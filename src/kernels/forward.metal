#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// vec_add
// ----------------------------------------------------------------
kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* result [[buffer(2)]],
                    constant int& n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < (uint)n) result[gid] = a[gid] + b[gid];
}

// ----------------------------------------------------------------
// matmul_forward_naive
// ----------------------------------------------------------------
kernel void matmul_forward_naive(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int bt = gid.x, oc = gid.y;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device float* wrow = weight + oc * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * wrow[i];
    out[bt * OC + oc] = val;
}

// ----------------------------------------------------------------
// softmax_forward
// ----------------------------------------------------------------
kernel void softmax_forward(
    device const float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int* p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    device float* out_row = out + row * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++)
        if (inp_row[j] > maxval) maxval = inp_row[j];

    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
        float v = exp(inp_row[j] - maxval);
        out_row[j] = v;
        sum += v;
    }
    float norm = 1.0f / sum;
    for (int j = 0; j < C; j++) out_row[j] *= norm;
}

// ----------------------------------------------------------------
// layernorm_forward
// ----------------------------------------------------------------
kernel void layernorm_forward(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    device float* out_row = out + row * C;

    float mean = 0.0f;
    for (int j = 0; j < C; j++) mean += inp_row[j];
    mean /= (float)C;

    float var = 0.0f;
    for (int j = 0; j < C; j++) {
        float d = inp_row[j] - mean;
        var += d * d;
    }
    var /= (float)C;

    float inv_std = 1.0f / sqrt(var + eps);
    for (int j = 0; j < C; j++) {
        float n = (inp_row[j] - mean) * inv_std;
        out_row[j] = n * weight[j] + bias[j];
    }
}

// ----------------------------------------------------------------
// rmsnorm_forward
// ----------------------------------------------------------------
kernel void rmsnorm_forward(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int* p [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    device float* out_row = out + row * C;

    float ss = 0.0f;
    for (int j = 0; j < C; j++) ss += inp_row[j] * inp_row[j];
    ss /= (float)C;

    float inv_rms = 1.0f / sqrt(ss + eps);
    for (int j = 0; j < C; j++)
        out_row[j] = inp_row[j] * inv_rms * weight[j];
}

// ----------------------------------------------------------------
// gelu_forward
// ----------------------------------------------------------------
kernel void gelu_forward(
    device const float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float x = inp[gid];
    float cdf = 0.5f * (1.0f + tanh(0.79788456f * x * (1.0f + 0.044715f * x * x)));
    out[gid] = x * cdf;
}

// ----------------------------------------------------------------
// swiglu_forward
// ----------------------------------------------------------------
kernel void swiglu_forward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    float x = gate[gid];
    float silu = x / (1.0f + exp(-x));
    out[gid] = silu * up[gid];
}

// ----------------------------------------------------------------
// encoder_forward
// ----------------------------------------------------------------
kernel void encoder_forward(
    device const float* wte [[buffer(0)]],
    device const float* wpe [[buffer(1)]],
    device const int* tokens [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int BT = p[0], C = p[1];
    int bt = (int)gid;
    if (bt >= BT) return;
    int token = tokens[bt];
    const device float* emb = wte + token * C;
    device float* out_bt = out + bt * C;
    for (int i = 0; i < C; i++) {
        float pos_add = wpe ? wpe[bt * C + i] : 0.0f;
        out_bt[i] = emb[i] + pos_add;
    }
}

// ----------------------------------------------------------------
// residual_forward
// ----------------------------------------------------------------
kernel void residual_forward(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= N) return;
    dst[gid] += src[gid];
}

// ----------------------------------------------------------------
// softmax_causal_scale
// ----------------------------------------------------------------
kernel void softmax_causal_scale(
    device float* scores [[buffer(0)]],
    constant int& seq_len [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int row = (int)gid;
    device float* srow = scores + row * seq_len;

    for (int t = 0; t < seq_len; t++) srow[t] *= scale;

    float maxval = -INFINITY;
    for (int t = 0; t <= row; t++) {
        if (srow[t] > maxval) maxval = srow[t];
    }

    float sum = 0.0f;
    for (int t = 0; t <= row; t++) {
        srow[t] = exp(srow[t] - maxval);
        sum += srow[t];
    }
    float norm = 1.0f / sum;
    for (int t = 0; t <= row; t++) srow[t] *= norm;
    for (int t = row + 1; t < seq_len; t++) srow[t] = 0.0f;
}

// ----------------------------------------------------------------
// rope_forward
// ----------------------------------------------------------------
kernel void rope_forward(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    constant int& head_size [[buffer(2)]],
    constant int& pos [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    device float* qh = q + h * head_size;
    device float* kh = k + h * head_size;

    for (int i = 0; i < head_size; i += 2) {
        float freq = 1.0f / pow(theta, (float)i / (float)head_size);
        float cosv = cos((float)pos * freq);
        float sinv = sin((float)pos * freq);

        float q0 = qh[i], q1 = qh[i+1];
        qh[i]   = q0 * cosv - q1 * sinv;
        qh[i+1] = q0 * sinv + q1 * cosv;

        float k0 = kh[i], k1 = kh[i+1];
        kh[i]   = k0 * cosv - k1 * sinv;
        kh[i+1] = k0 * sinv + k1 * cosv;
    }
}
