// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_MODEL_H
#define BMT_MODEL_H

#include <stddef.h>
#include "baremetal.h"

struct bm_model_s {
    bm_arch_t arch;
    bm_precision_t precision;

    size_t   n_parameters;

    int      kv_dim;
    int      head_size;
    int      n_kv_heads;
    int      kv_mul;

    void*    weight_buffer;

    float*   token_embedding_table;
    bm_bf16_t* token_embedding_bf16;
    float*   wpe;
    float*   ln1w;
    float*   ln1b;
    float*   qkvw;
    float*   qkvb;
    float*   qw;
    float*   kw;
    float*   vw;
    float*   attprojw;
    float*   attprojb;
    float*   ln2w;
    float*   ln2b;
    float*   fcw;
    float*   fcb;
    float*   fcw3;
    float*   fcprojw;
    float*   fcprojb;
    float*   lnfw;
    float*   lnfb;
    float*   wcls;
    float*   q_norm_w;
    float*   k_norm_w;
    float*   pre_ffn_w;
    float*   ffn_post_w;

    void*    graph;

    struct bmt_lora_adapter_s* lora;
    int      n_lora;
    float*   lora_ab;

    int      quant_bits;  /* 0 = fp, 8 = Q8, 4 = Q4 */

    int      weight_is_mmap;
    size_t   weight_map_size;
    struct bmt_weight_src_s* weight_srcs;
    int      n_weight_srcs;
    void*    stream_set;
    void     (*stream_close)(void*);
};

typedef struct bmt_weight_src_s {
    size_t     offset;
    const void* ptr;
    int        is_bf16;
} bmt_weight_src_t;

int  bmt_model_alloc_buffers(bm_model_t* model, const bm_arch_t* arch);
int  bmt_model_alloc_buffers_ex(bm_model_t* model, const bm_arch_t* arch, int use_mmap);
void bmt_model_free_buffers(bm_model_t* model);
void bmt_model_release_host_weights(bm_model_t* model);
int  bmt_model_load_weights(bm_model_t* model, const char* path);
int  bmt_arch_validate(const bm_arch_t* arch, int has_rope_scaling);

#endif
