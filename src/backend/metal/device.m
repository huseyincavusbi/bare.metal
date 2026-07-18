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

    BMT_LOG_INFO("Metal device: %s", [[device name] UTF8String]);
    return ctx;
}

void backend_destroy(backend_ctx_t* ctx) {
    if (!ctx) return;
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
    return buf;
}

void backend_buffer_free(backend_buffer_t* buf) {
    if (!buf) return;
    if (buf->buffer) {
        id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf->buffer;
        (void)b;
    }
    free(buf);
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
