// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_LORA_H
#define BMT_LORA_H

#include <stddef.h>

struct bm_model_s;

typedef enum {
    BMT_LORA_Q    = 1 << 0,
    BMT_LORA_K    = 1 << 1,
    BMT_LORA_V    = 1 << 2,
    BMT_LORA_O    = 1 << 3,
    BMT_LORA_GATE = 1 << 4,
    BMT_LORA_UP   = 1 << 5,
    BMT_LORA_DOWN = 1 << 6,
} bmt_lora_target_t;

#define BMT_LORA_ATTN   (BMT_LORA_Q | BMT_LORA_K | BMT_LORA_V | BMT_LORA_O)
#define BMT_LORA_MLP    (BMT_LORA_GATE | BMT_LORA_UP | BMT_LORA_DOWN)
#define BMT_LORA_ALL    (BMT_LORA_ATTN | BMT_LORA_MLP)

typedef struct {
    int   rank;
    float alpha;
    int   targets;
    int   seed;
} bmt_lora_config_t;

typedef struct bmt_lora_adapter_s {
    int   layer;
    int   target;
    int   r;
    int   c;
    int   oc;
    float scale;
    float* a;
    float* b;
    int   t_a;
    int   t_b;
} bmt_lora_adapter_t;

int  bmt_lora_init(struct bm_model_s* model, const bmt_lora_config_t* cfg);
void bmt_lora_free(struct bm_model_s* model);
bmt_lora_adapter_t* bmt_lora_find(struct bm_model_s* model, int layer, int target);
int  bmt_lora_parse_targets(const char* spec);

#endif
