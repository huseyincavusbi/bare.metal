// bare.metal forward pass kernels
// Placeholder — kernels will be ported from llm.c/dev/cuda/ in Phase 1

#include <metal_stdlib>
using namespace metal;

kernel void placeholder_kernel(device float* output [[buffer(0)]],
                                uint gid [[thread_position_in_grid]]) {
    output[gid] = 0.0f;
}
