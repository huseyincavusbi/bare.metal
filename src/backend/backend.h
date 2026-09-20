// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_BACKEND_H
#define BMT_BACKEND_H

#include "baremetal.h"
#include <stddef.h>

typedef struct backend_ctx_s     backend_ctx_t;
typedef struct backend_buffer_s  backend_buffer_t;
typedef struct backend_kernel_s  backend_kernel_t;
typedef struct backend_encoder_s backend_encoder_t;

backend_ctx_t*    backend_create(void);
void              backend_destroy(backend_ctx_t* ctx);
size_t            backend_get_allocated_memory(backend_ctx_t* ctx);
size_t            backend_get_peak_allocated_memory(backend_ctx_t* ctx);

typedef struct {
    int    max_threads_per_threadgroup;
    size_t max_threadgroup_memory;
    size_t max_buffer_bytes;
    size_t recommended_max_working_set;
    int    has_unified_memory;
} backend_device_info_t;

void              backend_get_device_info(backend_ctx_t* ctx, backend_device_info_t* out);

backend_buffer_t* backend_buffer_alloc(backend_ctx_t* ctx, size_t size);
void              backend_buffer_free(backend_buffer_t* buf);
void*             backend_buffer_map(backend_buffer_t* buf);
void              backend_buffer_unmap(backend_buffer_t* buf);
size_t            backend_buffer_size(backend_buffer_t* buf);

backend_kernel_t* backend_kernel_create(backend_ctx_t* ctx, const char* name);
void              backend_kernel_destroy(backend_kernel_t* kernel);

int               backend_kernel_dispatch(backend_ctx_t* ctx,
                       backend_kernel_t* kernel,
                       backend_buffer_t* buffers[],
                       size_t offsets[],
                       int num_buffers,
                       int grid_x, int grid_y, int grid_z,
                       int tg_x,  int tg_y,  int tg_z);
void              backend_synchronize(backend_ctx_t* ctx);

backend_encoder_t* backend_encode_begin(backend_ctx_t* ctx);
int               backend_encode_dispatch(backend_encoder_t* enc,
                       backend_kernel_t* kernel,
                       backend_buffer_t* buffers[],
                       size_t offsets[],
                       int num_buffers,
                       int grid_x, int grid_y, int grid_z,
                       int tg_x,  int tg_y,  int tg_z);
void              backend_encode_commit(backend_encoder_t* enc);
void              backend_encode_wait(backend_encoder_t* enc);

#endif
