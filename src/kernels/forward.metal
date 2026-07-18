#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* result [[buffer(2)]],
                    constant int& n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < (uint)n) {
        result[gid] = a[gid] + b[gid];
    }
}

kernel void matmul_forward_naive(
    device const float* inp [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int BT = params[0];
    int C  = params[1];
    int OC = params[2];
    int has_bias = params[3];

    int bt = gid.x;
    int oc = gid.y;
    if (bt >= BT || oc >= OC) return;

    float val = has_bias ? bias[oc] : 0.0f;
    const device float* inp_bt = inp + bt * C;
    const device float* wrow = weight + oc * C;
    for (int i = 0; i < C; i++) {
        val += inp_bt[i] * wrow[i];
    }
    out[bt * OC + oc] = val;
}
