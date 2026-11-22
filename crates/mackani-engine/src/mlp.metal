
// The MLP span between the gate/up and down projections, so one command
// buffer covers gate+up, the gated product, its int8 quantization and down.
// Each kernel performs the scalar reference's binary32 operations in order;
// there is no multiply-add that a contraction could fuse. The GPU flushes
// binary32 subnormals to zero, so every value whose result could differ
// from gradual underflow, and every non-finite value (which the reference
// rejects), sets `inexact`; the host then recomputes the span on the CPU.

constant float HADAMARD_NORMALIZATION = as_type<float>(0x3d3504f3u);
constant float INTEGER_SCALE_RANGE = 16777216.0f;
constant float INTEGER_SCALE_QUANTUM = as_type<float>(0x33800000u);
constant uint MLP_THREADS = 256;

struct MlpParams {
    uint gate_rows;
    uint rows;
    uint groups;
};

static uint exponent_bits(float value) {
    return (as_type<uint>(value) >> 23) & 0xffu;
}

static bool is_zero(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) == 0;
}

// Finite and not subnormal: the GPU uses it exactly as the CPU does.
static bool exact_operand(float value) {
    const uint exponent = exponent_bits(value);
    return exponent != 0xffu && (exponent != 0 || is_zero(value));
}

// At least 2^-103 in magnitude. Sums and differences of such values (and
// zeros) are multiples of 2^-126, so the butterflies never round to a
// subnormal.
static bool butterfly_operand(float value) {
    return exponent_bits(value) >= 24;
}

static float round_bfloat(float value) {
    const uint bits = as_type<uint>(value);
    return as_type<float>(((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16) << 16);
}

// The reference's ascending-stride butterflies over one 512-value block.
static void hadamard512(threadgroup float* values, uint lane) {
    for (uint stride = 1; stride < 512; stride <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lower = lane / stride * 2 * stride + lane % stride;
        const float a = values[lower];
        const float b = values[lower + stride];
        values[lower] = a + b;
        values[lower + stride] = a - b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// (row quantum, inverse RMS) restore of one projection sum.
static float restore(long sum, float2 row, float scale, thread bool& exact) {
    float value = float(sum) * row.x;
    value *= row.y;
    value *= scale;
    exact = exact && exact_operand(scale) && exact_operand(value) &&
            (sum == 0 || is_zero(row.x) || is_zero(row.y) || is_zero(scale) ||
             butterfly_operand(value));
    return value;
}

static float inverse_rotation(float rotated, char sign, thread bool& exact) {
    const float value = (rotated * HADAMARD_NORMALIZATION) * float(sign);
    exact = exact && exact_operand(value) && is_zero(rotated) == is_zero(value);
    return value;
}

// bfloat(bfloat(silu(bfloat(gate))) * bfloat(up)); the SiLU of each BF16
// gate comes from the reference through `silu`.
static float gated(float gate, float up, device const ushort* silu, thread bool& exact) {
    const float activation = as_type<float>(uint(silu[as_type<uint>(round_bfloat(gate)) >> 16]) << 16);
    const float rounded_up = round_bfloat(up);
    const float product = activation * rounded_up;
    const float output = round_bfloat(product);
    exact = exact && exact_operand(activation) && exact_operand(rounded_up) &&
            exact_operand(output) &&
            (is_zero(activation) || is_zero(rounded_up) || butterfly_operand(product));
    return output;
}

// One threadgroup restores, inverse-rotates and gates one 512-row block of
// one activation row; `product` is M x rows.
kernel void mlp_gated_product(
    device const long* gate_sums [[buffer(0)]],
    device const long* up_sums [[buffer(1)]],
    device const float* gate_scale [[buffer(2)]],
    device const float* up_scale [[buffer(3)]],
    device const char* gate_signs [[buffer(4)]],
    device const char* up_signs [[buffer(5)]],
    device const float2* row_inputs [[buffer(6)]],
    device const ushort* silu [[buffer(7)]],
    device float* product [[buffer(8)]],
    device atomic_uint* inexact [[buffer(9)]],
    constant MlpParams& params [[buffer(10)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float gate[512];
    threadgroup float up[512];
    const uint m = block.y;
    const uint first = block.x * 512;
    const float2 row = row_inputs[m];
    bool exact = exact_operand(row.x) && exact_operand(row.y);
    for (uint k = lane; k < 512; k += MLP_THREADS) {
        const uint r = first + k;
        const ulong at = ulong(m) * params.gate_rows + r;
        gate[k] = restore(gate_sums[at], row, gate_scale[r], exact);
        up[k] = restore(up_sums[at], row, up_scale[r], exact);
    }
    hadamard512(gate, lane);
    hadamard512(up, lane);
    for (uint k = lane; k < 512; k += MLP_THREADS) {
        const uint r = first + k;
        const float g = inverse_rotation(gate[k], gate_signs[r], exact);
        const float u = inverse_rotation(up[k], up_signs[r], exact);
        if (r < params.rows) {
            product[ulong(m) * params.rows + r] = gated(g, u, silu, exact);
        }
    }
    if (!exact) {
        atomic_fetch_or_explicit(inexact, 1u, memory_order_relaxed);
    }
}

// One threadgroup rotates and quantizes one 512-value group of one product
// row into exact binary16 integers and its activation scale.
kernel void mlp_quantize(
    device const float* product [[buffer(0)]],
    device const char* signs [[buffer(1)]],
    device half* activation [[buffer(2)]],
    device float* activation_scale [[buffer(3)]],
    device atomic_uint* inexact [[buffer(4)]],
    constant MlpParams& params [[buffer(5)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float values[512];
    threadgroup float maxima[MLP_THREADS / 32];
    const ulong row = ulong(block.y) * params.rows + block.x * 512;
    const device char* group_signs = signs + block.x * 512;
    for (uint k = lane; k < 512; k += MLP_THREADS) {
        values[k] = product[row + k] * float(group_signs[k]);
    }
    hadamard512(values, lane);
    bool exact = true;
    float maximum = 0.0f;
    for (uint k = lane; k < 512; k += MLP_THREADS) {
        const float scaled = values[k] * HADAMARD_NORMALIZATION;
        // From 65520 binary16 rounding overflows; the reference rejects it.
        exact = exact && isfinite(scaled) && fabs(scaled) < 65520.0f;
        values[k] = float(half(scaled));
        maximum = max(maximum, fabs(values[k]));
    }
    maximum = simd_max(maximum);
    if (simd_lane == 0) {
        maxima[simd_group] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < MLP_THREADS / 32; g++) {
        maximum = max(maximum, maxima[g]);
    }
    const float scale = precise::divide(maximum, 127.0f);
    for (uint k = lane; k < 512; k += MLP_THREADS) {
        const float ratio = scale > 0.0f ? precise::divide(values[k], scale) : 0.0f;
        activation[row + k] = half(char(clamp(rint(ratio), -127.0f, 127.0f)));
    }
    if (lane == 0) {
        activation_scale[ulong(block.y) * params.groups + block.x] = scale;
    }
    if (!exact) {
        atomic_fetch_or_explicit(inexact, 1u, memory_order_relaxed);
    }
}

// One thread per activation row: the shared 24-bit integer scale grid.
kernel void mlp_scales(
    device const float* activation_scale [[buffer(0)]],
    device int* integer_scale [[buffer(1)]],
    device float* row_quantum [[buffer(2)]],
    constant MlpParams& params [[buffer(3)]],
    uint m [[threadgroup_position_in_grid]])
{
    const ulong first = ulong(m) * params.groups;
    float maximum = 0.0f;
    for (uint g = 0; g < params.groups; g++) {
        maximum = max(maximum, activation_scale[first + g]);
    }
    const float multiplier = maximum > 0.0f ? precise::divide(INTEGER_SCALE_RANGE, maximum) : 0.0f;
    row_quantum[m] = maximum * INTEGER_SCALE_QUANTUM;
    for (uint g = 0; g < params.groups; g++) {
        integer_scale[first + g] = int(rint(activation_scale[first + g] * multiplier));
    }
}
