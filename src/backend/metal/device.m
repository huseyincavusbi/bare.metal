// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "utils/log.h"
#include <stdlib.h>

backend_ctx_t* backend_create(void) {
    backend_ctx_t* ctx = calloc(1, sizeof(backend_ctx_t));
    if (!ctx) return NULL;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        BMT_LOG_ERROR("No Metal device found");
        free(ctx);
        return NULL;
    }
    ctx->device = (__bridge_retained void*)device;
    ctx->queue  = (__bridge_retained void*)[device newCommandQueue];

    if (@available(macOS 13.0, *)) {
        ctx->supports_bf16 = [device supportsFamily:MTLGPUFamilyMetal3];
    } else {
        ctx->supports_bf16 = 0;
    }

    ctx->dev_max_threads_per_tg    = (int)[device maxThreadsPerThreadgroup].width;
    ctx->dev_max_threadgroup_mem   = (size_t)[device maxThreadgroupMemoryLength];
    ctx->dev_max_buffer_bytes      = (size_t)[device maxBufferLength];
    ctx->dev_recommended_working_set = (size_t)[device recommendedMaxWorkingSetSize];
    ctx->dev_has_unified_memory    = [device hasUnifiedMemory] ? 1 : 0;

    BMT_LOG_INFO("Metal device: %s (bf16: %s)",
                 [[device name] UTF8String],
                 ctx->supports_bf16 ? "yes" : "no");

    NSString* path = [[NSBundle mainBundle] pathForResource:@"default"
                                                     ofType:@"metallib"];
    if (!path) {
        NSString* exePath = [[NSProcessInfo processInfo] arguments][0];
        NSString* exeDir = [exePath stringByDeletingLastPathComponent];
        path = [exeDir stringByAppendingPathComponent:@"kernels/default.metallib"];
    }

    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:path];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) {
        BMT_LOG_WARN("Could not load metallib from %s: %s",
                     [path UTF8String],
                     [[error localizedDescription] UTF8String]);
    } else {
        ctx->library = (__bridge_retained void*)library;
        BMT_LOG_INFO("Loaded metallib");
    }

    return ctx;
}

void backend_destroy(backend_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->library) {
        id<MTLLibrary> l = (__bridge_transfer id<MTLLibrary>)ctx->library;
        (void)l;
    }
    if (ctx->queue) {
        id<MTLCommandQueue> q = (__bridge_transfer id<MTLCommandQueue>)ctx->queue;
        (void)q;
    }
    if (ctx->device) {
        id<MTLDevice> d = (__bridge_transfer id<MTLDevice>)ctx->device;
        (void)d;
    }
    free(ctx);
}

backend_buffer_t* backend_buffer_alloc(backend_ctx_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    id<MTLDevice> device = (__bridge id<MTLDevice>)ctx->device;
    id<MTLBuffer> buffer = [device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
    if (!buffer) {
        BMT_LOG_ERROR("Failed to allocate %zu bytes", size);
        return NULL;
    }

    backend_buffer_t* buf = calloc(1, sizeof(backend_buffer_t));
    buf->buffer = (__bridge_retained void*)buffer;
    buf->size   = size;
    buf->ctx    = ctx;
    ctx->allocated_bytes += size;
    if (ctx->allocated_bytes > ctx->peak_allocated_bytes)
        ctx->peak_allocated_bytes = ctx->allocated_bytes;
    return buf;
}

void backend_buffer_free(backend_buffer_t* buf) {
    if (!buf) return;
    if (buf->ctx && buf->ctx->allocated_bytes >= buf->size)
        buf->ctx->allocated_bytes -= buf->size;
    if (buf->buffer) {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf->buffer;
        (void)b;
    }
    free(buf);
}

size_t backend_get_allocated_memory(backend_ctx_t* ctx) {
    if (!ctx) return 0;
    return ctx->allocated_bytes;
}

size_t backend_get_peak_allocated_memory(backend_ctx_t* ctx) {
    if (!ctx) return 0;
    return ctx->peak_allocated_bytes;
}

void backend_get_device_info(backend_ctx_t* ctx, backend_device_info_t* out) {
    if (!ctx || !out) return;
    out->max_threads_per_threadgroup = ctx->dev_max_threads_per_tg;
    out->max_threadgroup_memory      = ctx->dev_max_threadgroup_mem;
    out->max_buffer_bytes            = ctx->dev_max_buffer_bytes;
    out->recommended_max_working_set = ctx->dev_recommended_working_set;
    out->has_unified_memory          = ctx->dev_has_unified_memory;
}

double backend_get_gpu_busy_ms(backend_ctx_t* ctx) {
    return ctx ? ctx->gpu_busy_ms : 0.0;
}

uint64_t backend_get_command_buffers(backend_ctx_t* ctx) {
    return ctx ? ctx->cmd_buffers : 0;
}

void backend_reset_gpu_timing(backend_ctx_t* ctx) {
    if (!ctx) return;
    ctx->gpu_busy_ms = 0.0;
    ctx->cmd_buffers = 0;
}

void* backend_buffer_map(backend_buffer_t* buf) {
    if (!buf || !buf->buffer) return NULL;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf->buffer;
    return [b contents];
}

void backend_buffer_unmap(backend_buffer_t* buf) {
    (void)buf;
}

size_t backend_buffer_size(backend_buffer_t* buf) {
    return buf ? buf->size : 0;
}
