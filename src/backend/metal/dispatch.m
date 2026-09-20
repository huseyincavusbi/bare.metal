// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
                            int grid_x, int grid_y, int grid_z,
                            int tg_x,  int tg_y,  int tg_z) {
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

    MTLSize gridSize = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize threadgroupSize = MTLSizeMake(tg_x, tg_y, tg_z);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    if (cmdBuf.GPUStartTime > 0.0 && cmdBuf.GPUEndTime >= cmdBuf.GPUStartTime)
        ctx->gpu_busy_ms += (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
    ctx->cmd_buffers += 1;

    return 0;
}

void backend_synchronize(backend_ctx_t* ctx) {
    (void)ctx;
}

backend_encoder_t* backend_encode_begin(backend_ctx_t* ctx) {
    if (!ctx) return NULL;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)ctx->queue;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    backend_encoder_t* e = calloc(1, sizeof(backend_encoder_t));
    e->command_buffer = (__bridge_retained void*)cmdBuf;
    e->encoder = (__bridge_retained void*)enc;
    e->ctx = ctx;
    return e;
}

int backend_encode_dispatch(backend_encoder_t* enc,
                            backend_kernel_t* kernel,
                            backend_buffer_t* buffers[],
                            size_t offsets[],
                            int num_buffers,
                            int grid_x, int grid_y, int grid_z,
                            int tg_x,  int tg_y,  int tg_z) {
    if (!enc || !kernel || !buffers) return -1;
    id<MTLComputeCommandEncoder> encoder =
        (__bridge id<MTLComputeCommandEncoder>)enc->encoder;
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)kernel->pipeline_state;

    [encoder setComputePipelineState:pso];
    for (int i = 0; i < num_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[i]->buffer;
        [encoder setBuffer:buf offset:offsets ? offsets[i] : 0 atIndex:i];
    }

    /* Per-kernel profiling: bracket the dispatch with GPU timestamp samples. */
    int profiled = 0;
    if (enc->ctx && enc->ctx->profiling && enc->ctx->profile_idx + 1 < enc->ctx->profile_n) {
        id<MTLCounterSampleBuffer> sb =
            (__bridge id<MTLCounterSampleBuffer>)enc->ctx->profile_buf;
        if (sb) {
            [encoder sampleCountersInBuffer:sb
                             atSampleIndex:(NSUInteger)enc->ctx->profile_idx++
                               withBarrier:YES];
            [encoder dispatchThreads:MTLSizeMake(grid_x,grid_y,grid_z)
               threadsPerThreadgroup:MTLSizeMake(tg_x,tg_y,tg_z)];
            [encoder sampleCountersInBuffer:sb
                             atSampleIndex:(NSUInteger)enc->ctx->profile_idx++
                               withBarrier:YES];
            profiled = 1;
        }
    }
    if (!profiled) {
        [encoder dispatchThreads:MTLSizeMake(grid_x,grid_y,grid_z)
           threadsPerThreadgroup:MTLSizeMake(tg_x,tg_y,tg_z)];
    }
    return 0;
}

void backend_encode_commit(backend_encoder_t* enc) {
    if (!enc) return;
    [(__bridge id<MTLComputeCommandEncoder>)enc->encoder endEncoding];
    [(__bridge id<MTLCommandBuffer>)enc->command_buffer commit];
}

void backend_encode_wait(backend_encoder_t* enc) {
    if (!enc) return;
    id<MTLCommandBuffer> cmdBuf =
        (__bridge id<MTLCommandBuffer>)enc->command_buffer;
    [cmdBuf waitUntilCompleted];
    if (enc->ctx) {
        if (cmdBuf.GPUStartTime > 0.0 && cmdBuf.GPUEndTime >= cmdBuf.GPUStartTime)
            enc->ctx->gpu_busy_ms += (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
        enc->ctx->cmd_buffers += 1;
    }
    id<MTLComputeCommandEncoder> encoder =
        (__bridge_transfer id<MTLComputeCommandEncoder>)enc->encoder;
    id<MTLCommandBuffer> released =
        (__bridge_transfer id<MTLCommandBuffer>)enc->command_buffer;
    (void)encoder; (void)released;
    free(enc);
}
