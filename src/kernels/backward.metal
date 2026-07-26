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
