// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
    int bt = gid.y, oc = gid.x;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device float* wrow = weight + oc * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * wrow[i];
    out[bt * OC + oc] = val;
}

// ----------------------------------------------------------------
// matmul_forward_bf16
// Same as naive but weights stored as bf16 (2x memory savings).
// Reads bf16, casts to float for computation, outputs float.
// ----------------------------------------------------------------
#if __HAVE_BFLOAT__
kernel void matmul_forward_bf16(
    device const float* inp [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int bt = gid.y, oc = gid.x;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device bfloat* wrow = weight + oc * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * (float)wrow[i];
    out[bt * OC + oc] = val;
}
#endif

// ----------------------------------------------------------------
// matmul_forward_fp16
// Same as naive but weights stored as fp16 (2x memory savings).
// Reads half, casts to float for computation, outputs float.
// ----------------------------------------------------------------
kernel void matmul_forward_fp16(
    device const float* inp [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int bt = gid.y, oc = gid.x;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device half* wrow = weight + oc * C;
    for (int i = 0; i < C; i++) val += inp_bt[i] * (float)wrow[i];
    out[bt * OC + oc] = val;
}

// ----------------------------------------------------------------
// matmul_forward_tiled
// Tiled GEMM with threadgroup shared memory for input reuse.
// One threadgroup computes BN output elements, reusing one shared
// input chunk (BK elements) across all BN threads. This eliminates
// redundant input reads: naive reads OC*C input; tiled reads (OC/BN)*C.
// Dispatch: grid=(ceil(OC/BN)*BN, BT, 1), threadgroup=(BN, 1, 1)
// ----------------------------------------------------------------
#define MM_BN 32
#define MM_BK 32

kernel void matmul_forward_tiled(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int tid = (int)tid3.x;

    int bt = (int)tgid.y;
    int oc_start = (int)tgid.x * MM_BN;
    int oc = oc_start + tid;
    if (bt >= BT || oc >= OC) return;

    threadgroup float A_s[MM_BK];
    threadgroup float B_s[MM_BN * MM_BK];

    float val = has_bias ? bias[oc] : 0.0f;

    for (int kk = 0; kk < C; kk += MM_BK) {
        int actual_bk = min(MM_BK, C - kk);

        for (int i = tid; i < actual_bk; i += MM_BN) {
            A_s[i] = inp[bt * C + kk + i];
        }

        for (int i = tid; i < MM_BN * actual_bk; i += MM_BN) {
            int n = i / actual_bk;
            int k = i % actual_bk;
            int woc = oc_start + n;
            if (woc < OC) {
                B_s[n * MM_BK + k] = weight[woc * C + kk + k];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < actual_bk; k++) {
            val += A_s[k] * B_s[tid * MM_BK + k];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[bt * OC + oc] = val;
}

// ----------------------------------------------------------------
// Q8_0 quantized matmul forward
// Weight tensor stored as q8_block_t blocks (fp32 scale + 32 int8),
// 36 bytes per 32 weights. Dequantizes on the fly: w = q * scale.
// Layout: weight[oc] is a row of (C/32) blocks. C must be a multiple of 32.
// Same thread/grid convention as the fp32 naive kernel: gid.x=oc, gid.y=bt.
// ----------------------------------------------------------------
typedef struct {
    float scale;
    char  q[32];
} q8_block_metal;  // 36 bytes — must match C q8_block_t layout

kernel void matmul_forward_q8(
    device const float* inp [[buffer(0)]],
    device const q8_block_metal* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int nblocks = C / 32;
    int bt = gid.y, oc = gid.x;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device q8_block_metal* wrow = weight + (size_t)oc * nblocks;
    for (int b = 0; b < nblocks; b++) {
        float s = wrow[b].scale;
        const device char* qb = wrow[b].q;
        const device float* xb = inp_bt + b * 32;
        for (int i = 0; i < 32; i++) {
            val += xb[i] * ((float)qb[i] * s);
        }
    }
    out[bt * OC + oc] = val;
}

// ----------------------------------------------------------------
// matmul_forward_q4  (Q4_0-style: 32 int4 weights + fp32 scale per block)
// ----------------------------------------------------------------
typedef struct {
    float   scale;
    uchar   qs[16];
} q4_block_metal;  // 20 bytes — must match C q4_block_t layout

kernel void matmul_forward_q4(
    device const float* inp [[buffer(0)]],
    device const q4_block_metal* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2], has_bias = params[3];
    int nblocks = C / 32;
    int bt = gid.y, oc = gid.x;
    if (bt >= BT || oc >= OC) return;
    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device q4_block_metal* wrow = weight + (size_t)oc * nblocks;
    for (int b = 0; b < nblocks; b++) {
        float s = wrow[b].scale;
        const device uchar* qb = wrow[b].qs;
        const device float* xb = inp_bt + b * 32;
        for (int j = 0; j < 16; j++) {
            uchar byte = qb[j];
            int lo = (int)(char)(byte << 4) >> 4;   // sign-extend low nibble
            int hi = (int)(char)byte >> 4;          // sign-extend high nibble
            val += xb[2 * j]     * ((float)lo * s);
            val += xb[2 * j + 1] * ((float)hi * s);
        }
    }
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
    device float* out [[buffer(2)]],
    constant int* p [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    const device float* bias_row = bias + row * C;
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
        out_row[j] = n * weight[j] + bias_row[j];
    }
}

// ----------------------------------------------------------------
// layernorm_forward_v2 - warp-reduced, single pass
// ----------------------------------------------------------------
kernel void layernorm_forward_v2(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int* p [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int N = p[0], C = p[1];
    int row = (int)tgid.x;
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    device float* out_row = out + row * C;

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += tptg) local_sum += inp_row[j];
    for (int off = 16; off > 0; off >>= 1)
        local_sum += simd_shuffle_down(local_sum, (ushort)off);

    threadgroup float shared_buf[32];
    if (tid % 32 == 0) shared_buf[tid / 32] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? shared_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) shared_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared_buf[0] / (float)C;

    float local_sq = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float d = inp_row[j] - mean;
        local_sq += d * d;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sq += simd_shuffle_down(local_sq, (ushort)off);
    if (tid % 32 == 0) shared_buf[tid / 32] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? shared_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) shared_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = shared_buf[0] / (float)C;
    float inv_std = 1.0f / sqrt(var + eps);

    for (int j = tid; j < C; j += tptg) {
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
// rmsnorm_forward_v2 - warp-reduced
// ----------------------------------------------------------------
kernel void rmsnorm_forward_v2(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int* p [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int N = p[0], C = p[1];
    int row = (int)tgid.x;
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;
    if (row >= N) return;
    const device float* inp_row = inp + row * C;
    device float* out_row = out + row * C;

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = inp_row[j];
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float shared_buf[32];
    if (tid % 32 == 0) shared_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? shared_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) shared_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = shared_buf[0] / (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    for (int j = tid; j < C; j += tptg)
        out_row[j] = inp_row[j] * inv_rms * weight[j];
}

// ----------------------------------------------------------------
// residual_rmsnorm_forward - fused residual add + RMSNorm
// out = rmsnorm(x + residual, weight, eps)
// Saves one D-element buffer roundtrip vs separate add + norm.
// Modifies x in-place to store y = x + residual (for inference efficiency).
// ----------------------------------------------------------------
kernel void residual_rmsnorm_forward(
    device float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int N = p[0], C = p[1];
    int row = (int)tgid.x;
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;
    if (row >= N) return;

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = x[row * C + j] + residual[row * C + j];
        x[row * C + j] = v;
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float shared_buf[32];
    if (tid % 32 == 0) shared_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? shared_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) shared_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = shared_buf[0] / (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    for (int j = tid; j < C; j += tptg)
        out[row * C + j] = x[row * C + j] * inv_rms * weight[j];
}

// ----------------------------------------------------------------
// residual_rmsnorm_forward_train
// Training-safe variant: does NOT modify x in-place.
// Computes y = x + residual on the fly, outputs rmsnorm(y, w).
// Backward kernels recompute y the same way (no stored y buffer).
// ----------------------------------------------------------------
kernel void residual_rmsnorm_forward_train(
    device const float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int N = p[0], C = p[1];
    int row = (int)tgid.x;
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;
    if (row >= N) return;

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = x[row * C + j] + residual[row * C + j];
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float shared_buf[32];
    if (tid % 32 == 0) shared_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? shared_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) shared_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = shared_buf[0] / (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    for (int j = tid; j < C; j += tptg) {
        float v = x[row * C + j] + residual[row * C + j];
        out[row * C + j] = v * inv_rms * weight[j];
    }
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
    constant int& n_kv_heads [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int h = (int)gid;
    device float* qh = q + h * head_size;
    int hd2 = head_size / 2;

    for (int i = 0; i < hd2; i++) {
        float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
        float cosv = cos((float)pos * freq);
        float sinv = sin((float)pos * freq);

        float q0 = qh[i], q1 = qh[i + hd2];
        qh[i]      = q0 * cosv - q1 * sinv;
        qh[i + hd2] = q0 * sinv + q1 * cosv;

        if (h < n_kv_heads) {
            device float* kh = k + h * head_size;
            float k0 = kh[i], k1 = kh[i + hd2];
            kh[i]      = k0 * cosv - k1 * sinv;
            kh[i + hd2] = k0 * sinv + k1 * cosv;
        }
    }
}

// ----------------------------------------------------------------
// attention_forward
// One threadgroup per query head. Within each threadgroup:
//   Phase 1: threads compute QK^T dot products (one per timestep)
//   Phase 2: thread 0 does softmax over scores
//   Phase 3: threads compute weighted V output (one per dimension)
// GQA: kh = h / kv_mul maps query head to KV head
// ----------------------------------------------------------------
kernel void attention_forward(
    device const float* q [[buffer(0)]],
    device const float* kv_cache [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int* params [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int NH = params[0], HD = params[1];
    int KV = params[3], S = params[4];
    int kv_mul = params[5], layer = params[6], MS = params[7];

    int h = (int)tgid.x;
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;
    if (h >= NH) return;

    int kh = h / kv_mul;
    int kv_off = layer * 2 * MS * KV;
    const device float* k_cache = kv_cache + kv_off;
    const device float* v_cache = kv_cache + kv_off + MS * KV;
    const device float* qh = q + h * HD;

    threadgroup float scores[1024];

    for (int t = tid; t < S; t += tptg) {
        float score = 0.0f;
        for (int i = 0; i < HD; i++) {
            score += qh[i] * k_cache[t * KV + kh * HD + i];
        }
        scores[t] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float reduce_buf[32];

    float local_max = -INFINITY;
    for (int t = tid; t < S; t += tptg) {
        if (scores[t] > local_max) local_max = scores[t];
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = simd_shuffle_down(local_max, (ushort)off);
        if (other > local_max) local_max = other;
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? reduce_buf[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            float other = simd_shuffle_down(v, (ushort)off);
            if (other > v) v = other;
        }
        if (tid == 0) reduce_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reduce_buf[0];

    float local_sum = 0.0f;
    for (int t = tid; t < S; t += tptg) {
        scores[t] = exp(scores[t] - global_max);
        local_sum += scores[t];
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += simd_shuffle_down(local_sum, (ushort)off);
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += simd_shuffle_down(v, (ushort)off);
        }
        if (tid == 0) reduce_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / reduce_buf[0];

    for (int t = tid; t < S; t += tptg) scores[t] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tid; i < HD; i += tptg) {
        float val = 0.0f;
        for (int t = 0; t < S; t++) {
            val += scores[t] * v_cache[t * KV + kh * HD + i];
        }
        out[h * HD + i] = val;
    }
}

// ----------------------------------------------------------------
// attention_forward_flash
// Online Flash Attention: one threadgroup per query head, each
// thread owns one output dimension. Iterates over K/V one key at a
// time, accumulating running max/sum and weighted V output without
// spilling scores to threadgroup memory. Tradeoff: each thread
// recomputes the score dot product, but scores never touch shared
// memory => better memory bandwidth and lower occupancy pressure.
// Dispatch: grid=(NH*HD,1,1), threadgroup=(HD,1,1)
// ----------------------------------------------------------------
kernel void attention_forward_flash(
    device const float* q [[buffer(0)]],
    device const float* kv_cache [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int* params [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    int NH = params[0], HD = params[1];
    int KV = params[3], S = params[4];
    int kv_mul = params[5], layer = params[6], MS = params[7];

    int h = (int)tgid.x;
    int i = (int)tid3.x;
    if (h >= NH || i >= HD) return;

    int kh = h / kv_mul;
    int kv_off = layer * 2 * MS * KV;
    const device float* k_cache = kv_cache + kv_off;
    const device float* v_cache = kv_cache + kv_off + MS * KV;
    const device float* qh = q + h * HD;

    threadgroup float q_shared[256];
    if (i < HD) q_shared[i] = qh[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int t = 0; t < S; t++) {
        float score = 0.0f;
        for (int j = 0; j < HD; j++) {
            score += q_shared[j] * k_cache[t * KV + kh * HD + j];
        }
        score *= scale;

        float new_max = fast::max(running_max, score);
        float exp_old = fast::exp(running_max - new_max);
        float exp_new = fast::exp(score - new_max);
        running_sum = running_sum * exp_old + exp_new;
        out_val = out_val * exp_old + exp_new * v_cache[t * KV + kh * HD + i];
        running_max = new_max;
    }

    out[h * HD + i] = out_val / running_sum;
}

// ----------------------------------------------------------------
// rmsnorm_matmul_forward - fused classifier
// Computes out = matmul(rmsnorm(inp, norm_weight, eps), wcls)
// Saves a D-element VRAM roundtrip.
// ----------------------------------------------------------------
kernel void rmsnorm_matmul_forward(
    device const float* inp [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* wcls [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int C = p[0], V = p[1];
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;

    threadgroup float shared_x[4096];

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = inp[j];
        shared_x[j] = v;
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float reduce_buf[32];
    if (tid % 32 == 0) reduce_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) reduce_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = 1.0f / sqrt((reduce_buf[0] / (float)C) + eps);

    for (int j = tid; j < C; j += tptg) {
        shared_x[j] = shared_x[j] * inv_rms * norm_weight[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int global_id = (int)tgid.x * tptg + tid;
    if (global_id >= V) return;

    const device float* wrow = wcls + global_id * C;
    float val = 0.0f;
    for (int j = 0; j < C; j++) {
        val += shared_x[j] * wrow[j];
    }
    out[global_id] = val;
}

#if __HAVE_BFLOAT__
kernel void rmsnorm_matmul_forward_bf16(
    device const float* inp [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const bfloat* wcls [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int C = p[0], V = p[1];
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;

    threadgroup float shared_x[4096];

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = inp[j];
        shared_x[j] = v;
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float reduce_buf[32];
    if (tid % 32 == 0) reduce_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) reduce_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = 1.0f / sqrt((reduce_buf[0] / (float)C) + eps);

    for (int j = tid; j < C; j += tptg) {
        shared_x[j] = shared_x[j] * inv_rms * norm_weight[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int global_id = (int)tgid.x * tptg + tid;
    if (global_id >= V) return;

    const device bfloat* wrow = wcls + global_id * C;
    float val = 0.0f;
    for (int j = 0; j < C; j++) {
        val += shared_x[j] * (float)wrow[j];
    }
    out[global_id] = val;
}
#endif

kernel void rmsnorm_matmul_forward_fp16(
    device const float* inp [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const half* wcls [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* p [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tptg3 [[threads_per_threadgroup]])
{
    int C = p[0], V = p[1];
    int tid = (int)tid3.x;
    int tptg = (int)tptg3.x;

    threadgroup float shared_x[4096];

    float local_ss = 0.0f;
    for (int j = tid; j < C; j += tptg) {
        float v = inp[j];
        shared_x[j] = v;
        local_ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_ss += simd_shuffle_down(local_ss, (ushort)off);

    threadgroup float reduce_buf[32];
    if (tid % 32 == 0) reduce_buf[tid / 32] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float v = (tid < (tptg + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += simd_shuffle_down(v, (ushort)off);
        if (tid == 0) reduce_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = 1.0f / sqrt((reduce_buf[0] / (float)C) + eps);

    for (int j = tid; j < C; j += tptg) {
        shared_x[j] = shared_x[j] * inv_rms * norm_weight[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int global_id = (int)tgid.x * tptg + tid;
    if (global_id >= V) return;

    const device half* wrow = wcls + global_id * C;
    float val = 0.0f;
    for (int j = 0; j < C; j++) {
        val += shared_x[j] * (float)wrow[j];
    }
    out[global_id] = val;
}
// attention_forward_seq (training forward: full sequence, causal, GQA)
// Counterpart to attention_backward. For each query pos i, head h (kh=h/kv_mul):
//   scores[j] = (Q[h,i] . K[kh,j]) * scale,  j in [0,i]   (causal)
//   p = softmax(scores);  out[h,i] = sum_j p[j] * V[kh,j]
// One thread per (h,i), loops over HD. Grid = NH*S.
// buffers: [0]=Q[NH,S,HD] [1]=K[NKV,S,HD] [2]=V [3]=out[NH,S,HD]
//          [4]=params[NH,S,HD,NKV,kv_mul]  [5]=scale
// ----------------------------------------------------------------
#define ATTN_MAXS_FWD 256
kernel void attention_forward_seq(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float* out       [[buffer(3)]],
    constant int* p         [[buffer(4)]],
    constant float& scale   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int NH = p[0], S = p[1], HD = p[2], NKV = p[3], kv_mul = p[4];
    int h = (int)gid / S;
    int i = (int)gid - h * S;
    if (h >= NH || i >= S) return;
    int kh = h / kv_mul;
    const device float* qi = Q + (i * NH + h) * HD;   /* [S,NH,HD] layout */

    thread float sc[ATTN_MAXS_FWD], pr[ATTN_MAXS_FWD];
    int n = i + 1;
    float mx = -INFINITY;
    for (int j = 0; j < n; j++) {
        const device float* kj = K + (j * NKV + kh) * HD;
        float s = 0.0f;
        for (int d = 0; d < HD; d++) s += qi[d] * kj[d];
        sc[j] = s * scale;
        if (sc[j] > mx) mx = sc[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < n; j++) { pr[j] = exp(sc[j] - mx); sum += pr[j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j < n; j++) pr[j] *= inv;

    device float* oi = out + (i * NH + h) * HD;
    for (int d = 0; d < HD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += pr[j] * V[(j * NKV + kh) * HD + d];
        oi[d] = acc;
    }
}

// ----------------------------------------------------------------
// rope_forward_seq (training forward RoPE, full sequence, [S,NH,HD] layout)
// For each (position s, head h): freq=1/theta^(2i/HD); apply rotation by pos=s.
//   qh[i]    = qh[i]*c - qh[i+hd2]*s ;  qh[i+hd2] = qh[i]*s + qh[i+hd2]*c
// In-place on q and k. One thread per (s,h). Grid = S*NH.
// buffers: [0]=q[S,NH,HD] [1]=k[S,NKV,HD]  [2]=params{head_size,n_kv_heads,S,NH}  [3]=theta
// ----------------------------------------------------------------
kernel void rope_forward_seq(
    device float* q            [[buffer(0)]],
    device float* k            [[buffer(1)]],
    constant int* p            [[buffer(2)]],
    constant float& theta      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int head_size = p[0], n_kv_heads = p[1], S = p[2], NH = p[3];
    int s = (int)gid / NH;
    int h = (int)gid - s * NH;
    if (s >= S || h >= NH) return;
    int hd2 = head_size / 2;
    device float* qh = q + (s * NH + h) * head_size;
    for (int i = 0; i < hd2; i++) {
        float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
        float c = cos((float)s * freq);
        float s_ = sin((float)s * freq);
        float q0 = qh[i], q1 = qh[i + hd2];
        qh[i]      = q0 * c - q1 * s_;
        qh[i + hd2] = q0 * s_ + q1 * c;
    }
    if (h < n_kv_heads) {
        device float* kh = k + (s * n_kv_heads + h) * head_size;
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / pow(theta, (float)(2*i) / (float)head_size);
            float c = cos((float)s * freq);
            float s_ = sin((float)s * freq);
            float k0 = kh[i], k1 = kh[i + hd2];
            kh[i]      = k0 * c - k1 * s_;
            kh[i + hd2] = k0 * s_ + k1 * c;
        }
    }
}

