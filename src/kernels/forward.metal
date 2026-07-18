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
