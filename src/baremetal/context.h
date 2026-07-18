#ifndef BMT_CONTEXT_H
#define BMT_CONTEXT_H

#include "backend/backend.h"

struct bm_context_s {
    bm_device_t    device_type;
    backend_ctx_t* backend_ctx;
};

bm_context_t* bmt_context_create(bm_device_t device);
void          bmt_context_destroy(bm_context_t* ctx);

#endif
