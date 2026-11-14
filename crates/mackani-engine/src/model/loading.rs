//! Mapping boundary and validated main-model assembly.
use super::*;

impl Model {
    /// Load with detected hardware, Int8 KV storage and 16 rows per pass.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: the caller guarantees file stability until this model is dropped.
        unsafe {
            Self::with_options(
                path,
                Engine::detected()?,
                Backend::detected(),
                16,
                KvMode::Int8,
            )
        }
    }

    /// The CPU backend with an explicit kernel/pool and pass width (positive, with
    /// no implicit upper clamp).
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_engine(
        path: impl AsRef<Path>,
        engine: Engine,
        rows_per_pass: usize,
    ) -> Result<Self> {
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_backend(path, engine, Backend::Cpu, rows_per_pass) }
    }

    /// An explicit projection backend. `engine` runs all other row/head work, and
    /// the projections too when `backend` is [`Backend::Cpu`]. Backend selection is
    /// explicit. Metal fails without a device.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_backend(
        path: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        rows_per_pass: usize,
    ) -> Result<Self> {
        // SAFETY: forwarded file-stability contract is required of this method's caller.
        unsafe { Self::with_options(path, engine, backend, rows_per_pass, KvMode::Int8) }
    }

    /// Load with an explicit backend, pool, pass width and KV mode.
    /// Profile modes require a model folder containing the manifest's KVQP profile.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_options(
        path: impl AsRef<Path>,
        engine: Engine,
        backend: Backend,
        rows_per_pass: usize,
        kv_mode: KvMode,
    ) -> Result<Self> {
        // SAFETY: the caller guarantees file stability for the returned model.
        unsafe {
            Self::with_execution_options(
                path,
                engine,
                ExecutionOptions {
                    backend,
                    ..ExecutionOptions::default()
                },
                rows_per_pass,
                kv_mode,
            )
        }
    }

    /// Load mapped weights with explicit execution choices, pass width and KV mode.
    /// # Safety
    /// All weights files must remain unmodified and untruncated until this model is dropped.
    pub unsafe fn with_execution_options(
        path: impl AsRef<Path>,
        engine: Engine,
        options: ExecutionOptions,
        rows_per_pass: usize,
        kv_mode: KvMode,
    ) -> Result<Self> {
        if rows_per_pass == 0 {
            return Err(Error::InvalidInput("rows_per_pass must be positive".into()));
        }
        let mut engine = Compute::new(engine, options.backend)?;
        engine.cpu_stages = options.cpu_stages.fast();
        engine.set_metal_options(options.metal);
        let (path, context_capacity, profile) = model_location_with_kv(path.as_ref(), kv_mode)?;
        // SAFETY: caller establishes file stability through model drop. Every weight view
        // shares this mapping, so none can outlive that external-file contract.
        let mapping = unsafe { MappedModel::open(path)? };
        let weights = load_weights_with_kv(
            &mut mapping.reader()?,
            kv_mode,
            profile,
            |_, name, rows, cols, output_rotation| {
                let matrix = mapping.matrix(name)?;
                let d = matrix.descriptor();
                if d.rows != rows || d.cols != cols {
                    return Err(Error::InvalidModel(format!(
                        "{name} must have shape [{rows}, {cols}]"
                    )));
                }
                if output_rotation
                    && (d.physical_rows % 512 != 0 || matrix.output_signs().is_none())
                {
                    return Err(Error::InvalidModel(
                        "body projection requires complete output rotations".into(),
                    ));
                }
                Ok(Linear {
                    matrix,
                    output_rotation,
                })
            },
            |_, name| mapping.embedding_tensor(name),
        )?;
        let states = weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        let working = weights
            .layers
            .iter()
            .map(Layer::new_state)
            .collect::<Result<_>>()?;
        let replay = weights
            .layers
            .iter()
            .map(|layer| match &layer.branch {
                Branch::Attention(_) => Ok(Vec::new()),
                Branch::Recurrent(layer) => (0..rows_per_pass)
                    .map(|_| layer.recurrence.new_row())
                    .collect::<Result<Vec<_>>>(),
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            working,
            replay,
            pending_tokens: Vec::with_capacity(rows_per_pass),
            pending: false,
            working_complete: false,
            context_identity: Arc::new(()),
            capture_options: CaptureOptions::default(),
            captures: Vec::new(),
            pending_captures: Vec::new(),
            committed_logits: None,
            tokens: Vec::new(),
            owner: Arc::new(()),
            weights,
            engine,
            states,
            position: 0,
            context_capacity,
            rows_per_pass,
            kv_mode,
            logits: Vec::new(),
            timings: Vec::new(),
            detailed_timing: false,
        })
    }
}
