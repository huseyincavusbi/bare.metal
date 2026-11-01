// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/lora.h"
#include "baremetal/scheduler.h"
#include "baremetal/trainer.h"
#include "utils/log.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BMLORA_MAGIC   0x424D4C52u
#define BMLORA_VERSION 1u

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  dim;
    int32_t  layers;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  vocab;
    int32_t  head_size;
    int32_t  rank;
    float    alpha;
    int32_t  n_adapters;
} bmlora_header_t;

typedef struct {
    int32_t layer;
    int32_t target;
    int32_t r;
    int32_t c;
    int32_t oc;
} bmlora_entry_t;

static float* base_projection(bm_model_t* model, int layer, int target, int* c, int* oc) {
    int D = model->arch.dim;
    int H = model->arch.hidden_dim;
    int NH = model->arch.n_heads;
    int HD = model->head_size;
    int KV = model->kv_dim;
    switch (target) {
        case BMT_LORA_Q:
            *c = D; *oc = NH * HD;
            return model->qw + (size_t)layer * NH * HD * D;
        case BMT_LORA_K:
            *c = D; *oc = KV;
            return model->kw + (size_t)layer * KV * D;
        case BMT_LORA_V:
            *c = D; *oc = KV;
            return model->vw + (size_t)layer * KV * D;
        case BMT_LORA_O:
            *c = NH * HD; *oc = D;
            return model->attprojw + (size_t)layer * NH * HD * D;
        case BMT_LORA_GATE:
            if (!model->arch.gated_mlp) return NULL;
            *c = D; *oc = H;
            return model->fcw + (size_t)layer * H * D;
        case BMT_LORA_UP:
            *c = D; *oc = H;
            if (model->arch.gated_mlp)
                return model->fcw3 + (size_t)layer * H * D;
            return model->fcw + (size_t)layer * H * D;
        case BMT_LORA_DOWN:
            *c = H; *oc = D;
            return model->fcprojw + (size_t)layer * D * H;
        default:
            return NULL;
    }
}

#ifdef BAREMETAL_TRAIN

static bmt_adamw_state_t* find_state(bm_trainer_t* t, int tensor_id) {
    for (int i = 0; i < t->n_opt_states; i++) {
        if (t->opt_states[i].grad_tensor_id == tensor_id) return &t->opt_states[i];
    }
    return NULL;
}

int bm_save_adapter(bm_trainer_t* trainer, const char* path) {
    if (!trainer || !path || trainer->model->n_lora == 0) return -1;
    bm_model_t* m = trainer->model;

    bmlora_header_t h;
    memset(&h, 0, sizeof(h));
    h.magic = BMLORA_MAGIC;
    h.version = BMLORA_VERSION;
    h.dim = m->arch.dim;
    h.layers = m->arch.n_layers;
    h.n_heads = m->arch.n_heads;
    h.n_kv_heads = m->arch.n_kv_heads;
    h.vocab = m->arch.vocab_size;
    h.head_size = m->head_size;
    h.rank = m->lora[0].r;
    h.alpha = m->lora[0].scale * (float)m->lora[0].r;
    h.n_adapters = m->n_lora;

    FILE* f = fopen(path, "wb");
    if (!f) {
        BMT_LOG_ERROR("bm_save_adapter: cannot open %s", path);
        return -1;
    }
    if (fwrite(&h, sizeof(h), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    for (int i = 0; i < m->n_lora; i++) {
        bmt_lora_adapter_t* a = &m->lora[i];
        bmt_adamw_state_t* sa = find_state(trainer, a->t_a);
        bmt_adamw_state_t* sb = find_state(trainer, a->t_b);
        if (!sa || !sb) {
            BMT_LOG_ERROR("bm_save_adapter: missing adapter state");
            fclose(f);
            return -1;
        }
        bmlora_entry_t e = { a->layer, a->target, a->r, a->c, a->oc };
        if (fwrite(&e, sizeof(e), 1, f) != 1) {
            fclose(f);
            return -1;
        }
        float* aw = sa->master_w ? (float*)backend_buffer_map(sa->master_w)
                                 : (float*)backend_buffer_map(bmt_scheduler_get_buffer(trainer->sched, a->t_a));
        float* bw = sb->master_w ? (float*)backend_buffer_map(sb->master_w)
                                 : (float*)backend_buffer_map(bmt_scheduler_get_buffer(trainer->sched, a->t_b));
        size_t an = (size_t)a->r * (size_t)a->c;
        size_t bn = (size_t)a->oc * (size_t)a->r;
        int ok = fwrite(aw, sizeof(float), an, f) == an &&
                 fwrite(bw, sizeof(float), bn, f) == bn;
        if (sa->master_w) backend_buffer_unmap(sa->master_w);
        else backend_buffer_unmap(bmt_scheduler_get_buffer(trainer->sched, a->t_a));
        if (sb->master_w) backend_buffer_unmap(sb->master_w);
        else backend_buffer_unmap(bmt_scheduler_get_buffer(trainer->sched, a->t_b));
        if (!ok) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

#endif /* BAREMETAL_TRAIN */

int bm_load_adapter(bm_model_t* model, const char* path) {
    if (!model || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) {
        BMT_LOG_ERROR("bm_load_adapter: cannot open %s", path);
        return -1;
    }
    bmlora_header_t h;
    if (fread(&h, sizeof(h), 1, f) != 1 ||
        h.magic != BMLORA_MAGIC || h.version != BMLORA_VERSION) {
        BMT_LOG_ERROR("bm_load_adapter: bad header");
        fclose(f);
        return -1;
    }
    if (h.dim != model->arch.dim || h.layers != model->arch.n_layers ||
        h.n_heads != model->arch.n_heads || h.n_kv_heads != model->arch.n_kv_heads ||
        h.vocab != model->arch.vocab_size || h.head_size != model->head_size ||
        h.rank <= 0 || h.n_adapters <= 0) {
        BMT_LOG_ERROR("bm_load_adapter: architecture mismatch");
        fclose(f);
        return -1;
    }

    float scale = h.alpha / (float)h.rank;
    for (int i = 0; i < h.n_adapters; i++) {
        bmlora_entry_t e;
        if (fread(&e, sizeof(e), 1, f) != 1 || e.r != h.rank || e.c <= 0 || e.oc <= 0) {
            BMT_LOG_ERROR("bm_load_adapter: bad entry");
            fclose(f);
            return -1;
        }
        int c, oc;
        float* w = base_projection(model, e.layer, e.target, &c, &oc);
        if (!w || c != e.c || oc != e.oc) {
            BMT_LOG_ERROR("bm_load_adapter: target mismatch");
            fclose(f);
            return -1;
        }
        size_t an = (size_t)e.r * (size_t)e.c;
        size_t bn = (size_t)e.oc * (size_t)e.r;
        float* a = (float*)malloc(an * sizeof(float));
        float* b = (float*)malloc(bn * sizeof(float));
        if (!a || !b || fread(a, sizeof(float), an, f) != an || fread(b, sizeof(float), bn, f) != bn) {
            BMT_LOG_ERROR("bm_load_adapter: short read");
            free(a); free(b); fclose(f);
            return -1;
        }
        for (int row = 0; row < e.oc; row++) {
            for (int col = 0; col < e.c; col++) {
                float sum = 0.0f;
                for (int k = 0; k < e.r; k++)
                    sum += b[(size_t)row * e.r + k] * a[(size_t)k * e.c + col];
                w[(size_t)row * e.c + col] += scale * sum;
            }
        }
        free(a); free(b);
    }
    fclose(f);
    return 0;
}
