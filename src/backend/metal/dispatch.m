#import <Metal/Metal.h>
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "utils/log.h"
#include <stdlib.h>

backend_kernel_t* backend_kernel_create(backend_ctx_t* ctx, const char* name) {
    if (!ctx || !ctx->library || !name) return NULL;

    id<MTLLibrary> library = (__bridge id<MTLLibrary>)ctx->library;
    id<MTLFunction> func = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!func) {
        BMT_LOG_ERROR("Kernel not found: %s", name);
        return NULL;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:func error:&error];
    if (!pso) {
        BMT_LOG_ERROR("Failed to create pipeline for %s: %s",
                     name, [[error localizedDescription] UTF8String]);
        return NULL;
    }

    backend_kernel_t* kernel = calloc(1, sizeof(backend_kernel_t));
    kernel->pipeline_state = (__bridge_retained void*)pso;
    BMT_LOG_INFO("Pipeline created: %s", name);
    return kernel;
}

void backend_kernel_destroy(backend_kernel_t* kernel) {
    if (!kernel) return;
    if (kernel->pipeline_state) {
        id<MTLComputePipelineState> pso =
            (__bridge_transfer id<MTLComputePipelineState>)kernel->pipeline_state;
        (void)pso;
    }
    free(kernel);
}

int backend_kernel_dispatch(backend_ctx_t* ctx,
                            backend_kernel_t* kernel,
                            backend_buffer_t* buffers[],
                            size_t offsets[],
                            int num_buffers,
                            int threadgroups,
                            int threads_per_threadgroup) {
    if (!ctx || !kernel || !buffers) return -1;

    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)kernel->pipeline_state;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pso];

    for (int i = 0; i < num_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[i]->buffer;
        [encoder setBuffer:buf offset:offsets ? offsets[i] : 0 atIndex:i];
    }

    MTLSize gridSize = MTLSizeMake(threadgroups * threads_per_threadgroup, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(threads_per_threadgroup, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return 0;
}

void backend_synchronize(backend_ctx_t* ctx) {
    (void)ctx;
}
