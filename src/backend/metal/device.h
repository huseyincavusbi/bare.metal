// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_METAL_DEVICE_H
#define BMT_METAL_DEVICE_H

struct backend_ctx_s {
    void* device;
    void* queue;
    void* library;
    int   supports_bf16;
    size_t allocated_bytes;       /* current live buffer bytes */
    size_t peak_allocated_bytes;  /* high-water mark */

    /* device limits captured at creation */
    int    dev_max_threads_per_tg;
    size_t dev_max_threadgroup_mem;
    size_t dev_max_buffer_bytes;
    size_t dev_recommended_working_set;
    int    dev_has_unified_memory;
};

struct backend_buffer_s {
    void* buffer;
    size_t size;
    struct backend_ctx_s* ctx;    /* owner, for accounting on free */
};

struct backend_kernel_s {
    void* pipeline_state;
};

struct backend_encoder_s {
    void* command_buffer;
    void* encoder;
};

#endif
