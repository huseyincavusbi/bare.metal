#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// matmul_backward_inp
// Forward: out[bt,oc] = bias[oc] + sum_i inp[bt,i] * w[oc,i]
// Backward wrt input:  grad_inp[bt,i] = sum_oc gout[bt,oc] * w[oc,i]
// Grid: (C, BT) — one thread per (bt, i). Same gid convention as the
// forward naive kernel (gid.x = inner dim index, gid.y = batch index).
// params: [0]=BT [1]=C [2]=OC
// ----------------------------------------------------------------
kernel void matmul_backward_inp(
    device const float* gout [[buffer(0)]],   // [BT, OC]
    device const float* w    [[buffer(1)]],   // [OC, C]
    device float* ginp       [[buffer(2)]],   // [BT, C]  (must be zeroed)
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int bt = gid.y, i = gid.x;
    if (bt >= BT || i >= C) return;
    float acc = 0.0f;
    for (int oc = 0; oc < OC; oc++) {
        acc += gout[bt * OC + oc] * w[oc * C + i];
    }
    ginp[bt * C + i] = acc;
}

// ----------------------------------------------------------------
// matmul_backward_w
// Backward wrt weight:  grad_w[oc,i] = sum_bt gout[bt,oc] * inp[bt,i]
// Grid: (C, OC) — one thread per (oc, i).
// params: [0]=BT [1]=C [2]=OC
// ----------------------------------------------------------------
kernel void matmul_backward_w(
    device const float* gout [[buffer(0)]],   // [BT, OC]
    device const float* inp  [[buffer(1)]],   // [BT, C]
    device float* gw         [[buffer(2)]],   // [OC, C]  (must be zeroed)
    constant int* params     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0], C = params[1], OC = params[2];
    int oc = gid.y, i = gid.x;
    if (oc >= OC || i >= C) return;
    float acc = 0.0f;
    for (int bt = 0; bt < BT; bt++) {
        acc += gout[bt * OC + oc] * inp[bt * C + i];
    }
    gw[oc * C + i] = acc;
}

// ----------------------------------------------------------------
// rmsnorm_backward_x
// Forward: ss = sum(x^2)/C ; inv_rms = 1/sqrt(ss+eps) ; n[j]=x[j]*inv_rms
//          out[j] = n[j]*w[j]
// Backward wrt input (per row, row length C):
//   c1 = (1/C) * sum_j gout[j]*out[j]
//   grad_x[j] = inv_rms * (gout[j]*w[j] - n[j]*c1)
// One thread per row (matches forward rmsnorm naive convention).
// buffers: [0]=gout[N,C] [1]=w[C] [2]=x[N,C] [3]=gx[N,C]  params[4]=[N,C]  eps[5]
// ----------------------------------------------------------------
kernel void rmsnorm_backward_x(
    device const float* gout [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* x    [[buffer(2)]],
    device float* gx         [[buffer(3)]],
    constant int* p          [[buffer(4)]],
    constant float& eps      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int row = (int)gid;
    if (row >= N) return;
    const device float* xr = x + row * C;
    const device float* gr = gout + row * C;
    device float* gxr = gx + row * C;

    float ss = 0.0f;
    for (int j = 0; j < C; j++) ss += xr[j] * xr[j];
    ss /= (float)C;
    float inv_rms = 1.0f / sqrt(ss + eps);

    // c1 = (1/C) * sum_j gout[j]*out[j], with out[j] = x[j]*inv_rms*w[j]
    float c1 = 0.0f;
    for (int j = 0; j < C; j++) c1 += gr[j] * xr[j] * inv_rms * w[j];
    c1 /= (float)C;

    for (int j = 0; j < C; j++) {
        float n = xr[j] * inv_rms;
        gxr[j] = inv_rms * (gr[j] * w[j] - n * c1);
    }
}

// ----------------------------------------------------------------
// rmsnorm_backward_w
// Backward wrt weight:  grad_w[j] = sum_row gout[row,j]*n[row,j]
//   where n[row,j] = x[row,j]*inv_rms(row)
// One thread per column j; loops over N rows accumulating. gw must be zeroed.
// buffers: [0]=gout[N,C] [1]=x[N,C] [2]=gw[C]  params[3]=[N,C]  eps[4]
// ----------------------------------------------------------------
kernel void rmsnorm_backward_w(
    device const float* gout [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float* gw         [[buffer(2)]],
    constant int* p          [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int N = p[0], C = p[1];
    int j = (int)gid;
    if (j >= C) return;
    float acc = 0.0f;
    for (int row = 0; row < N; row++) {
        const device float* xr = x + row * C;
        float ss = 0.0f;
        for (int k = 0; k < C; k++) ss += xr[k] * xr[k];
        ss /= (float)C;
        float inv_rms = 1.0f / sqrt(ss + eps);
        float n = xr[j] * inv_rms;
        acc += gout[row * C + j] * n;
    }
    gw[j] = acc;
}
