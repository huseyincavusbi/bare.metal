#ifndef BMT_BACKEND_H
#define BMT_BACKEND_H

#include "baremetal.h"
#include <stddef.h>

typedef struct backend_ctx_s     backend_ctx_t;
typedef struct backend_buffer_s  backend_buffer_t;

backend_ctx_t*    backend_create(void);
void              backend_destroy(backend_ctx_t* ctx);

backend_buffer_t* backend_buffer_alloc(backend_ctx_t* ctx, size_t size);
void              backend_buffer_free(backend_buffer_t* buf);
void*             backend_buffer_map(backend_buffer_t* buf);
void              backend_buffer_unmap(backend_buffer_t* buf);
size_t            backend_buffer_size(backend_buffer_t* buf);

#endif
