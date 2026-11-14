//! Scalar draft primitives with fixed evaluation order.
use super::*;

pub(super) fn linear(w: &CompressedLinear, x: &[f32], signs: &[i8]) -> Result<Vec<f32>> {
    w.project(&quantize_activation(x, &signs[..x.len()])?, None)
}
pub(super) fn linear_rows(w: &CompressedLinear, x: &[f32], signs: &[i8]) -> Result<Vec<f32>> {
    let mut output = Vec::new();
    for row in x.chunks_exact(w.cols()) {
        output.extend(linear(w, row, signs)?);
    }
    Ok(output)
}
/// Normalize one draft row with fixed RMS reduction and rounding.
pub fn norm(x: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let mut partial = [0.0_f32; 256];
    for (lane, acc) in partial.iter_mut().enumerate() {
        for i in (lane..x.len()).step_by(256) {
            // q35_bdraft.cu:465 (unfused launch equivalent :34).
            *acc = x[i].mul_add(x[i], *acc);
        }
    }
    let mut total = 0.0;
    for warp in partial.chunks_exact(32) {
        total += xor_sum32(warp);
    }
    if !total.is_finite() {
        return Err(Error::NonFinite("draft norm squares"));
    }
    let r = rsqrt(total / x.len() as f32 + epsilon);
    let y: Vec<_> = x.iter().zip(weight).map(|(&x, &w)| (r * x) * w).collect();
    require_finite(&y, "draft norm output")?;
    Ok(y)
}
/// Normalize each draft row independently in row order.
pub fn norm_rows(x: &[f32], w: &[f32], e: f32) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    for row in x.chunks_exact(w.len()) {
        out.extend(norm(row, w, e)?);
    }
    Ok(out)
}
/// Normalize and rotate each draft head at an absolute position.
pub fn rope(x: &mut [f32], w: &[f32], position: usize, epsilon: f32) -> Result<()> {
    for head in x.chunks_exact_mut(HEAD_DIM) {
        let squares: Vec<_> = head.iter().map(|&v| v * v).collect();
        let g: Vec<_> = squares.chunks_exact(32).map(xor_sum32).collect();
        let sum = ((g[0] + g[1]) + g[2]) + g[3];
        if !sum.is_finite() {
            return Err(Error::NonFinite("draft RoPE squares"));
        }
        let r = rsqrt(sum / 128.0 + epsilon);
        let n: [f32; HEAD_DIM] = std::array::from_fn(|i| (r * head[i]) * w[i]);
        require_finite(&n, "draft head norm")?;
        for i in 0..64 {
            let frequency = exp2((-((2 * i) as f32) / 128.0) * 23.253_496_f32);
            let (cs, sn) = sincos(position as f32 * frequency)?;
            head[i] = n[i] * cs - n[i + 64] * sn;
            head[i + 64] = n[i + 64] * cs + n[i] * sn;
        }
    }
    require_finite(x, "draft RoPE")
}
/// Apply the row-dependent four-tap draft filter in source order.
pub fn dynamic_filter(
    x: &[f32],
    dynamic: &[f32],
    base: &[f32],
    width: usize,
    branch: usize,
) -> Vec<f32> {
    let groups = width / 16;
    let dw = 4 * groups;
    let mut y = vec![0.0; x.len()];
    for row in 0..x.len() / width {
        for c in 0..width {
            let g = c / 16;
            let d = row * dw + branch * 2 * groups;
            let mut value = (base[2 * branch * width + c] + dynamic[d + g]) * x[row * width + c];
            if row > 0 {
                value += (base[(2 * branch + 1) * width + c] + dynamic[d + groups + g])
                    * x[(row - 1) * width + c];
            }
            y[row * width + c] = value;
        }
    }
    y
}
/// SiLU-gated product, elementwise: `(g / (1 + exp(-g))) * u`.
pub fn glu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + exp(-g))) * u)
        .collect()
}
/// The reference's 32-lane XOR reduction tree, shared with fast drafts.
pub fn xor_sum32(values: &[f32]) -> f32 {
    crate::gated_deltanet::xor_sum32(values)
}
/// Add a projected draft residual in place with a BF16 output boundary.
pub fn add(x: &mut [f32], y: &[f32]) -> Result<()> {
    for (x, &y) in x.iter_mut().zip(y) {
        *x += y;
    }
    require_finite(x, "draft residual")
}

#[derive(Clone)]
struct Partial {
    maximum: f32,
    denominator: f32,
    output: [f32; HEAD_DIM],
}
impl Partial {
    fn empty() -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            denominator: 0.0,
            output: [0.0; HEAD_DIM],
        }
    }
}
/// Attend a complete draft block over its sliding history and future block rows.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rk: &[u16],
    rv: &[u16],
    p: usize,
    d: &DraftDimensions,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0; BLOCK_ROWS * d.query_heads * HEAD_DIM];
    let start = p.saturating_sub(d.window - 1);
    let count = p - start;
    let chunk = count.div_ceil(32);
    for row in 0..BLOCK_ROWS {
        for head in 0..d.query_heads {
            let kv = head / (d.query_heads / d.kv_heads);
            let query = &q[(row * d.query_heads + head) * HEAD_DIM
                ..(row * d.query_heads + head + 1) * HEAD_DIM];
            let mut parts = vec![Partial::empty(); 33];
            for (split, part) in parts[..32].iter_mut().enumerate() {
                let first = split * chunk;
                let last = count.min(first + chunk);
                let n = last.saturating_sub(first);
                let mut scores = [f32::NEG_INFINITY; 64];
                for (j, score) in scores[..n].iter_mut().enumerate() {
                    let absolute = start + first + j;
                    if absolute < (p + row).saturating_sub(d.window - 1) {
                        continue;
                    }
                    let offset = (kv * d.window + absolute % d.window) * HEAD_DIM;
                    let mut acc = 0.0_f32;
                    for channel in 0..HEAD_DIM {
                        // Nested pair FMAs execute even then odd: q35_bdraft.cu:175.
                        acc = query[channel]
                            .mul_add(mackani_format::bf16_to_f32(rk[offset + channel]), acc);
                    }
                    *score = acc * SCALE;
                    if !score.is_finite() {
                        return Err(Error::NonFinite("draft history score"));
                    }
                }
                // Max is exact for finite inputs; its pair/XOR tree cannot change it.
                part.maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if part.maximum == f32::NEG_INFINITY {
                    continue;
                }
                let weights: [f32; 64] = std::array::from_fn(|i| {
                    if scores[i] == f32::NEG_INFINITY {
                        0.0
                    } else {
                        exp(scores[i] - part.maximum)
                    }
                });
                let pairs: [f32; 32] = std::array::from_fn(|i| weights[i] + weights[i + 32]);
                part.denominator = xor_sum32(&pairs);
                for (j, &weight) in weights[..n].iter().enumerate() {
                    let offset = (kv * d.window + (start + first + j) % d.window) * HEAD_DIM;
                    for (c, output) in part.output.iter_mut().enumerate() {
                        // q35_bdraft.cu:205.
                        *output =
                            weight.mul_add(mackani_format::bf16_to_f32(rv[offset + c]), *output);
                    }
                }
            }
            let current = &mut parts[32];
            let scaled: [f32; HEAD_DIM] = std::array::from_fn(|c| query[c] * SCALE);
            for j in 0..BLOCK_ROWS {
                let offset = (j * d.kv_heads + kv) * HEAD_DIM;
                let mut lanes = [0.0; 32];
                for (lane, local) in lanes.iter_mut().enumerate() {
                    let c = 4 * lane;
                    // Three explicit nested FMAs, last product separate: q35_bdraft.cu:141.
                    *local = scaled[c].mul_add(
                        k[offset + c],
                        scaled[c + 1].mul_add(
                            k[offset + c + 1],
                            scaled[c + 2]
                                .mul_add(k[offset + c + 2], scaled[c + 3] * k[offset + c + 3]),
                        ),
                    );
                }
                let score = xor_sum32(&lanes);
                if !score.is_finite() {
                    return Err(Error::NonFinite("draft block score"));
                }
                let maximum = current.maximum.max(score);
                let carry = exp(current.maximum - maximum);
                let weight = exp(score - maximum);
                // q35_bdraft.cu:146,148.
                current.denominator = current.denominator.mul_add(carry, weight);
                for (c, output) in current.output.iter_mut().enumerate() {
                    *output = output.mul_add(carry, weight * v[offset + c]);
                }
                current.maximum = maximum;
            }
            let maximum = parts
                .iter()
                .map(|p| p.maximum)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            let mut output = [0.0; HEAD_DIM];
            for part in parts {
                if part.maximum == f32::NEG_INFINITY {
                    continue;
                }
                let weight = exp(part.maximum - maximum);
                // q35_bdraft.cu:579 (unfused-output equivalent :226).
                denominator = part.denominator.mul_add(weight, denominator);
                for (c, o) in output.iter_mut().enumerate() {
                    *o = part.output[c].mul_add(weight, *o);
                }
            }
            for (c, &o) in output.iter().enumerate() {
                out[(row * d.query_heads + head) * HEAD_DIM + c] = o / denominator;
            }
        }
    }
    require_finite(&out, "draft attention output")?;
    Ok(out)
}
