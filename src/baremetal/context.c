#include "baremetal/context.h"
#include "utils/log.h"
#include <stdlib.h>

bm_context_t* bmt_context_create(bm_device_t device) {
    bm_context_t* ctx = calloc(1, sizeof(bm_context_t));
    if (!ctx) return NULL;

    ctx->device_type = device;

    if (device == BM_DEVICE_METAL) {
        ctx->backend_ctx = backend_create();
        if (!ctx->backend_ctx) {
            free(ctx);
            return NULL;
        }
    }

    BMT_LOG_INFO("Context created (device=%d)", device);
    return ctx;
}

void bmt_context_destroy(bm_context_t* ctx) {
    if (!ctx) return;
    if (ctx->backend_ctx) {
        backend_destroy(ctx->backend_ctx);
    }
    free(ctx);
}
