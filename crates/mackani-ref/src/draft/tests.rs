use super::*;
fn dims() -> DraftDimensions {
    DraftDimensions {
        hidden: 512,
        features: 2560,
        feed_forward: 512,
        vocabulary: 32,
        layers: 1,
        query_heads: 4,
        kv_heads: 4,
        window: 8,
        correction_rank: 32,
        epsilon: 1e-6,
    }
}
#[test]
fn draft_attention_every_row_sees_future_block_rows_and_sliding_history() {
    let d = dims();
    let q = vec![0.0; 8 * 4 * HEAD_DIM];
    let k = q.clone();
    let mut v = q.clone();
    for h in 0..4 {
        v[(7 * 4 + h) * HEAD_DIM..(7 * 4 + h + 1) * HEAD_DIM].fill(8.0);
    }
    let rk = vec![0; 4 * 8 * HEAD_DIM];
    let mut rv = rk.clone();
    for h in 0..4 {
        for slot in 0..8 {
            rv[(h * 8 + slot) * HEAD_DIM..(h * 8 + slot + 1) * HEAD_DIM]
                .fill(mackani_format::f32_to_bf16(2.0));
        }
    }
    let out = attention(&q, &k, &v, &rk, &rv, 0, &d).unwrap();
    // Uniform weights: future row 7 contributes 8/8 to ALL rows, including 0.
    assert!(out.iter().all(|&x| x == 1.0));
    let out = attention(&q, &k, &v, &rk, &rv, 9, &d).unwrap();
    for row in 0..8 {
        let history = 7 - row;
        let expected = (8.0 + 2.0 * history as f32) / (8 + history) as f32;
        assert!(
            out[row * 4 * HEAD_DIM..(row + 1) * 4 * HEAD_DIM]
                .iter()
                .all(|&x| x.to_bits() == expected.to_bits())
        );
    }
}
#[test]
fn draft_rope_position_zero_preserves_head_channels_after_direct_weight_norm() {
    let mut x = vec![1.0; HEAD_DIM];
    let w = vec![2.0; HEAD_DIM];
    rope(&mut x, &w, 0, 1e-6).unwrap();
    let expected = 2.0 * rsqrt(1.0 + 1e-6);
    assert!(x.iter().all(|&v| v.to_bits() == expected.to_bits()));
}
#[test]
fn draft_rope_nonzero_position_matches_hand_rotation_and_head_weights() {
    let mut x = vec![1.0; HEAD_DIM];
    let mut weights = vec![0.0; HEAD_DIM];
    weights[0] = 2.0;
    weights[64] = 3.0;
    weights[32] = 4.0;
    weights[96] = 5.0;
    rope(&mut x, &weights, 1, 1e-6).unwrap();
    // RMS(x)=1; r=1/sqrt(1+1e-6). For pair 0, angle=1:
    // r*(2*cos(1)-3*sin(1), 3*cos(1)+2*sin(1)). Pair 32 has
    // angle=10^(-7/2), with direct head weights (4,5). Constants are
    // independently evaluated in binary64 from those two equations.
    for (channel, expected) in [
        (0, -1.443_807_620_783_779_8_f64),
        (64, 3.303_847_235_297_007_4_f64),
        (32, 3.998_416_661_988_438_5_f64),
        (96, 5.001_262_160_412_533_f64),
    ] {
        assert!(
            (f64::from(x[channel]) - expected).abs() < 3e-6,
            "channel={channel} got={} expected={expected}",
            x[channel]
        );
    }
    assert!(
        x.iter()
            .enumerate()
            .all(|(channel, &value)| { [0, 32, 64, 96].contains(&channel) || value == 0.0 })
    );
}

#[test]
fn draft_grouped_attention_nonuniform_scores_use_inverse_sqrt_head_scaling() {
    let mut d = dims();
    d.query_heads = 8;
    d.kv_heads = 2;
    let mut q = vec![0.0; BLOCK_ROWS * d.query_heads * HEAD_DIM];
    for row in 0..BLOCK_ROWS {
        for head in 0..d.query_heads {
            q[(row * d.query_heads + head) * HEAD_DIM] = (head + 1) as f32;
        }
    }
    let mut k = vec![0.0; BLOCK_ROWS * d.kv_heads * HEAD_DIM];
    let mut v = k.clone();
    let mut rk = vec![0; d.kv_heads * d.window * HEAD_DIM];
    let mut rv = rk.clone();
    for kv in 0..d.kv_heads {
        k[kv * HEAD_DIM] = (kv + 1) as f32;
        v[kv * HEAD_DIM] = (16 + kv * 32) as f32;
        rk[kv * d.window * HEAD_DIM] = mackani_format::f32_to_bf16(-((kv + 1) as f32));
        rv[kv * d.window * HEAD_DIM] = mackani_format::f32_to_bf16((8 + kv * 32) as f32);
    }
    let out = attention(&q, &k, &v, &rk, &rv, 1, &d).unwrap();
    for row in 0..BLOCK_ROWS {
        for head in 0..d.query_heads {
            // Eight query heads share two KV heads in contiguous groups of four.
            // Among current rows only key 0 has score s=q*k/sqrt(128),
            // the other seven scores are zero. History position 0 has -s
            // and is in the sliding window until block row 7.
            let kv = head / 4;
            let score = ((head + 1) * (kv + 1)) as f64 / 128.0_f64.sqrt();
            let current_weight = score.exp();
            let history_weight = if row < 7 { (-score).exp() } else { 0.0 };
            let expected = (current_weight * (16 + kv * 32) as f64
                + history_weight * (8 + kv * 32) as f64)
                / (current_weight + 7.0 + history_weight);
            let offset = (row * d.query_heads + head) * HEAD_DIM;
            assert!(
                (f64::from(out[offset]) - expected).abs() < 8e-6,
                "row={row} head={head}: {} != {expected}",
                out[offset]
            );
            assert!(out[offset + 1..offset + HEAD_DIM].iter().all(|&x| x == 0.0));
        }
    }
}

#[test]
fn draft_filter_uses_current_dynamic_previous_unmodified_row_and_branch_bases() {
    let x: Vec<_> = (0..3).flat_map(|r| vec![(r + 1) as f32; 512]).collect();
    let mut dynamic = vec![0.0; 3 * 128];
    dynamic[128 + 32] = 3.0;
    dynamic[2 * 128 + 64] = 4.0;
    let base: Vec<_> = [1.0, 2.0, 5.0, 7.0]
        .into_iter()
        .flat_map(|n| vec![n; 512])
        .collect();
    let pre = dynamic_filter(&x, &dynamic, &base, 512, 0);
    assert_eq!((pre[0], pre[512], pre[1024]), (1.0, 7.0, 7.0));
    let post = dynamic_filter(&x, &dynamic, &base, 512, 1);
    assert_eq!((post[0], post[512], post[1024]), (5.0, 17.0, 41.0));
}
