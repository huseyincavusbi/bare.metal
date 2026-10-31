// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_CONFIG_H
#define BMT_CONFIG_H

#include "baremetal.h"

typedef enum {
    BMT_ARCH_FAMILY_UNKNOWN = 0,
    BMT_ARCH_FAMILY_LLAMA,
    BMT_ARCH_FAMILY_QWEN2,
    BMT_ARCH_FAMILY_QWEN3,
    BMT_ARCH_FAMILY_GEMMA,
    BMT_ARCH_FAMILY_GPT2,
    BMT_ARCH_FAMILY_PHI3,
    BMT_ARCH_FAMILY_MISTRAL,
    BMT_ARCH_FAMILY_MIXTRAL,
} bmt_arch_family_t;

typedef enum {
    BMT_ARCH_BUILDER_GENERIC = 0,
    BMT_ARCH_BUILDER_CUSTOM,
} bmt_arch_builder_t;

#define BMT_WEIGHT_TRANSFORM_NONE        0u
#define BMT_WEIGHT_TRANSFORM_GEMMA_NORM  (1u << 0)
#define BMT_WEIGHT_TRANSFORM_EMBED_SCALE (1u << 1)
#define BMT_WEIGHT_TRANSFORM_LOGIT_CAP   (1u << 2)
#define BMT_WEIGHT_TRANSFORM_ROPE_INTERLEAVED (1u << 3)

typedef struct {
    const char* embed;
    const char* layers;
    const char* input_norm;
    const char* post_attn_norm;
    const char* q;
    const char* k;
    const char* v;
    const char* o;
    const char* gate;
    const char* up;
    const char* down;
    const char* final_norm;
    const char* lm_head;
    const char* q_norm;
    const char* k_norm;
    const char* pre_ffn_norm;
    const char* post_ffn_norm;
} bmt_tensor_map_t;

typedef struct {
    const char* name;
    bmt_arch_family_t family;
    const bmt_tensor_map_t* tensors;
    bmt_arch_builder_t builder;
    unsigned weight_transforms;
    int supported;
    const char* unsupported_reason;
} bmt_arch_spec_t;

typedef struct {
    char model_type[64];
    char architecture[64];

    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    int max_position_embeddings;
    int head_dim;

    int attention_bias;
    int mlp_bias;
    int tie_word_embeddings;

    int has_rms_norm_eps;
    int has_rope_theta;
    int has_rope_scaling;
    int rope_scaling_type;   /* 1 = llama3, 0 = none, -1 = unsupported */

    float rope_theta;
    float rope_factor;
    float rope_low_freq_factor;
    float rope_high_freq_factor;
    float rope_original_max;
    char hidden_act[32];
} bmt_config_t;

int  bmt_config_load(const char* dir_path, bmt_config_t* cfg);
const bmt_arch_spec_t* bmt_arch_spec_for_config(const bmt_config_t* cfg);
int  bmt_config_to_arch(const bmt_config_t* cfg, const bmt_arch_spec_t* spec,
                        bm_arch_t* arch);

#endif
