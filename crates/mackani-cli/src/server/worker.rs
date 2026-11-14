use super::{
    Config,
    admission::{Charge, waited_too_long},
    handle,
    response::HttpBody,
};
use crate::{
    Error, Result,
    chat::{ChatRunner, Event},
};
use bytes::Bytes;
use hyper::{Request, Response, body::Body};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc, oneshot};

pub(super) struct Cancel(pub(super) Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub(super) struct Job {
    pub(super) params: crate::chat::ChatParams,
    pub(super) ready: oneshot::Sender<Result<mpsc::Receiver<Result<Event>>>>,
    pub(super) cancel: Arc<AtomicBool>,
    /// End of the wait for the engine (`Limits::queue_timeout`), which lasts
    /// until the prompt is prepared and generation starts.
    pub(super) deadline: std::time::Instant,
    // Released when the engine thread drops the job, after its generation.
    pub(super) _charge: Charge,
}
pub(super) struct Shared {
    pub(super) cfg: Config,
    pub(super) budget: Arc<Semaphore>,
    pub(super) loaded: AtomicBool,
    pub(super) capacity: AtomicUsize,
    pub(super) shutdown: AtomicBool,
    pub(super) failure: Mutex<Option<Error>>,
    pub(super) failure_ready: tokio::sync::Notify,
}

/// Reusable HTTP service. The factory crosses the thread boundary; the engine
/// never does. Dropping the service cancels generation and joins its owner.
pub struct Api {
    pub(super) shared: Arc<Shared>,
    pub(super) jobs: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Api {
    pub fn new<F>(cfg: Config, factory: F) -> Result<Self>
    where
        F: FnOnce() -> Result<ChatRunner> + Send + 'static,
    {
        cfg.validate()?;
        let (jobs, mut queue) = mpsc::channel::<Job>(cfg.limits.max_connections);
        let shared = Arc::new(Shared {
            budget: Arc::new(Semaphore::new(cfg.limits.max_pending_bytes)),
            cfg,
            loaded: AtomicBool::new(false),
            capacity: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            failure: Mutex::new(None),
            failure_ready: tokio::sync::Notify::new(),
        });
        let state = shared.clone();
        let worker = std::thread::Builder::new()
            .name("mackani-engine".into())
            .spawn(move || {
                // A panic would leave /health answering 200 with no engine
                // behind it, so the process ends instead.
                struct ExitOnPanic;
                impl Drop for ExitOnPanic {
                    fn drop(&mut self) {
                        if std::thread::panicking() {
                            eprintln!("the engine thread panicked; exiting");
                            std::process::exit(101);
                        }
                    }
                }
                let _exit_on_panic = ExitOnPanic;
                let mut runner = match factory() {
                    Ok(mut r) => {
                        r.max_prompt_bytes = state.cfg.limits.max_prompt_bytes;
                        r
                    }
                    Err(e) => {
                        *state
                            .failure
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(e);
                        state.failure_ready.notify_one();
                        return;
                    }
                };
                state
                    .capacity
                    .store(runner.engine.context_capacity(), Ordering::Release);
                state.loaded.store(true, Ordering::Release);
                while let Some(job) = queue.blocking_recv() {
                    let shared = &state;
                    let cancelled = || {
                        job.cancel.load(Ordering::Acquire)
                            || shared.shutdown.load(Ordering::Acquire)
                    };
                    if cancelled() || job.ready.is_closed() {
                        continue;
                    }
                    // Between the preparation steps: a request whose client
                    // left, or whose wait has run out, is not prepared further.
                    let check_preparation = || {
                        if cancelled() || job.ready.is_closed() {
                            Err(Error::engine("the request was cancelled"))
                        } else if std::time::Instant::now() >= job.deadline {
                            Err(waited_too_long())
                        } else {
                            Ok(())
                        }
                    };
                    let prepared =
                        match runner.prepare_interruptible(&job.params, &check_preparation) {
                            Ok(p) => p,
                            Err(e) => {
                                let _ = job.ready.send(Err(e));
                                continue;
                            }
                        };
                    let (events, rx) = mpsc::channel(8);
                    if job.ready.send(Ok(rx)).is_err() {
                        continue;
                    }
                    let send = |mut event: Result<Event>| -> Result<()> {
                        let deadline = std::time::Instant::now() + shared.cfg.limits.stream_timeout;
                        loop {
                            if cancelled() {
                                return Err(Error::engine("client disconnected"));
                            }
                            match events.try_send(event) {
                                Ok(()) => return Ok(()),
                                Err(mpsc::error::TrySendError::Closed(_)) => {
                                    return Err(Error::engine("client disconnected"));
                                }
                                Err(mpsc::error::TrySendError::Full(e)) => event = e,
                            }
                            if std::time::Instant::now() >= deadline {
                                return Err(Error::engine("response consumer timed out"));
                            }
                            std::thread::sleep(Duration::from_millis(2));
                        }
                    };
                    let result =
                        runner.run_prepared(&job.params, prepared, &cancelled, &mut |event| {
                            send(Ok(event))
                        });
                    if let Err(mut e) = result {
                        e.message = e.message.chars().take(500).collect();
                        let _ = events.try_send(Err(e));
                    }
                }
            })
            .map_err(Error::engine)?;
        Ok(Self {
            shared,
            jobs: Some(jobs),
            worker: Some(worker),
        })
    }
    pub fn is_ready(&self) -> bool {
        self.shared.loaded.load(Ordering::Acquire)
    }
    pub async fn call<B>(&self, r: Request<B>) -> Response<HttpBody>
    where
        B: Body<Data = Bytes> + Unpin,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        handle(r, self).await
    }
}
impl Drop for Api {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        drop(self.jobs.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
