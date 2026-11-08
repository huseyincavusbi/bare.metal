// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_SFT_H
#define BMT_SFT_H

#include "baremetal/tokenizer.h"
#include <stdint.h>

typedef enum {
    BMT_SFT_CHATML = 0,
    BMT_SFT_LLAMA3 = 1,
} bmt_sft_template_t;

typedef struct {
    char* prompt;
    char* completion;
} bmt_sft_pair_t;

typedef struct {
    bmt_sft_pair_t* items;
    int n;
} bmt_sft_dataset_t;

int  bmt_sft_parse_template(const char* name);
int  bmt_sft_detect_template(const char* model_dir);
int  bmt_sft_load_jsonl(const char* path, bmt_sft_template_t tmpl, int max_examples,
                        bmt_sft_dataset_t* out, int* n_skipped);
void bmt_sft_free_dataset(bmt_sft_dataset_t* d);

int  bmt_sft_build_example(bm_tokenizer_t* tok, const bmt_sft_pair_t* pair, int S,
                           int* inputs, int* targets, uint8_t* attn, uint8_t* loss);
int  bmt_sft_render_chat(bmt_sft_template_t tmpl, const char* system,
                         const char* user, bm_tokenizer_t* tok,
                         char** out_prompt, int* out_stop_id);

#endif
