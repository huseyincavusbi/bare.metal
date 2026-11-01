// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/lora.h"
#include "baremetal/model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static unsigned int lora_rng(unsigned int* s) {
    unsigned int x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float lora_uniform(unsigned int* s, float lo, float hi) {
    float u = (float)(lora_rng(s) & 0x00FFFFFFu) / 16777216.0f;
    return lo + (hi - lo) * u;
}

static int target_dims(const bm_model_t* m, int target, int* c, int* oc) {
    int D = m->arch.dim;
    int H = m->arch.hidden_dim;
    int NH = m->arch.n_heads;
    int HD = m->head_size;
    int KV = m->kv_dim;
    switch (target) {
        case BMT_LORA_Q:    *c = D;       *oc = NH * HD; return 1;
        case BMT_LORA_K:
        case BMT_LORA_V:    *c = D;       *oc = KV;      return 1;
        case BMT_LORA_O:    *c = NH * HD; *oc = D;       return 1;
        case BMT_LORA_GATE: if (!m->arch.gated_mlp) return 0;
                            *c = D;       *oc = H;       return 1;
        case BMT_LORA_UP:   *c = D;       *oc = H;       return 1;
        case BMT_LORA_DOWN: *c = H;       *oc = D;       return 1;
        default: return 0;
    }
}

void bmt_lora_free(bm_model_t* model) {
    if (!model) return;
    free(model->lora);
    model->lora = NULL;
    free(model->lora_ab);
    model->lora_ab = NULL;
    model->n_lora = 0;
}

int bmt_lora_init(bm_model_t* model, const bmt_lora_config_t* cfg) {
    if (!model || !cfg || cfg->rank <= 0 || cfg->targets == 0) {
        bmt_lora_free(model);
        return 0;
    }

    int L = model->arch.n_layers;
    int n = 0;
    for (int l = 0; l < L; l++) {
        for (int bit = 0; bit < 7; bit++) {
            int target = 1 << bit;
            if (!(cfg->targets & target)) continue;
            int c, oc;
            if (target_dims(model, target, &c, &oc)) n++;
        }
    }
    if (n == 0) return 0;

    bmt_lora_free(model);
    model->lora = (bmt_lora_adapter_t*)calloc((size_t)n, sizeof(bmt_lora_adapter_t));
    if (!model->lora) return -1;

    size_t total = 0;
    for (int l = 0; l < L; l++) {
        for (int bit = 0; bit < 7; bit++) {
            int target = 1 << bit;
            if (!(cfg->targets & target)) continue;
            int c, oc;
            if (!target_dims(model, target, &c, &oc)) continue;
            total += (size_t)cfg->rank * (size_t)c;
            total += (size_t)oc * (size_t)cfg->rank;
        }
    }
    model->lora_ab = (float*)calloc(total, sizeof(float));
    if (!model->lora_ab) {
        bmt_lora_free(model);
        return -1;
    }

    unsigned int seed = (unsigned int)cfg->seed;
    if (seed == 0) seed = 1u;

    int idx = 0;
    float* cursor = model->lora_ab;
    for (int l = 0; l < L; l++) {
        for (int bit = 0; bit < 7; bit++) {
            int target = 1 << bit;
            if (!(cfg->targets & target)) continue;
            int c, oc;
            if (!target_dims(model, target, &c, &oc)) continue;

            bmt_lora_adapter_t* a = &model->lora[idx++];
            a->layer = l;
            a->target = target;
            a->r = cfg->rank;
            a->c = c;
            a->oc = oc;
            a->scale = cfg->alpha / (float)cfg->rank;
            a->a = cursor;
            cursor += (size_t)a->r * (size_t)c;
            a->b = cursor;
            cursor += (size_t)oc * (size_t)a->r;
            a->t_a = -1;
            a->t_b = -1;

            float bound = 1.0f / sqrtf((float)c);
            for (size_t i = 0; i < (size_t)a->r * (size_t)c; i++)
                a->a[i] = lora_uniform(&seed, -bound, bound);
            for (size_t i = 0; i < (size_t)oc * (size_t)a->r; i++)
                a->b[i] = 0.0f;
        }
    }
    model->n_lora = n;
    return 0;
}

bmt_lora_adapter_t* bmt_lora_find(bm_model_t* model, int layer, int target) {
    if (!model) return NULL;
    for (int i = 0; i < model->n_lora; i++) {
        if (model->lora[i].layer == layer && model->lora[i].target == target)
            return &model->lora[i];
    }
    return NULL;
}

int bmt_lora_parse_targets(const char* spec) {
    if (!spec || !spec[0]) return BMT_LORA_ALL;
    if (strcmp(spec, "all") == 0) return BMT_LORA_ALL;
    if (strcmp(spec, "attn") == 0) return BMT_LORA_ATTN;
    if (strcmp(spec, "mlp") == 0) return BMT_LORA_MLP;

    int mask = 0;
    const char* p = spec;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        const char* q = p;
        while (*q && *q != ',' && *q != ' ' && *q != '\t') q++;
        size_t len = (size_t)(q - p);
        if (len == 1 && p[0] == 'q') mask |= BMT_LORA_Q;
        else if (len == 1 && p[0] == 'k') mask |= BMT_LORA_K;
        else if (len == 1 && p[0] == 'v') mask |= BMT_LORA_V;
        else if (len == 1 && p[0] == 'o') mask |= BMT_LORA_O;
        else if (len == 4 && strncmp(p, "gate", 4) == 0) mask |= BMT_LORA_GATE;
        else if (len == 2 && strncmp(p, "up", 2) == 0) mask |= BMT_LORA_UP;
        else if (len == 4 && strncmp(p, "down", 4) == 0) mask |= BMT_LORA_DOWN;
        else if (len == 3 && strncmp(p, "mlp", 3) == 0) mask |= BMT_LORA_MLP;
        else if (len == 4 && strncmp(p, "attn", 4) == 0) mask |= BMT_LORA_ATTN;
        else if (len == 3 && strncmp(p, "all", 3) == 0) mask |= BMT_LORA_ALL;
        p = q;
    }
    return mask;
}
