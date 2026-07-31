// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/sampler.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { float val; int idx; } bmt_fi_t;

static int bmt_fi_desc(const void* a, const void* b) {
    float fa = ((const bmt_fi_t*)a)->val;
    float fb = ((const bmt_fi_t*)b)->val;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

static void bmt_softmax_inplace(float* x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* The same top-k/top-p multinomial sampler that drove the validated
 * inference path in run.c: temperature -> softmax -> top-k mask ->
 * top-p (nucleus) mask -> sample by inverse-CDF with an LCG. */
int bm_sample(bm_sampler_t* s, const float* logits) {
    if (!s || !logits) return 0;
    int n = s->vocab_size;
    if (s->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < n; i++) s->probs[i] = logits[i] / s->temperature;
    bmt_softmax_inplace(s->probs, n);

    if (s->top_k > 0 && s->top_k < n) {
        bmt_fi_t* sorted = (bmt_fi_t*)malloc(n * sizeof(bmt_fi_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = s->probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(bmt_fi_t), bmt_fi_desc);
            float thresh = sorted[s->top_k - 1].val;
            for (int i = 0; i < n; i++) if (s->probs[i] < thresh) s->probs[i] = 0.0f;
            free(sorted);
            float sum = 0.0f;
            for (int i = 0; i < n; i++) sum += s->probs[i];
            if (sum > 0.0f) for (int i = 0; i < n; i++) s->probs[i] /= sum;
        }
    }
    if (s->topp > 0.0f && s->topp < 1.0f) {
        bmt_fi_t* sorted = (bmt_fi_t*)malloc(n * sizeof(bmt_fi_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = s->probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(bmt_fi_t), bmt_fi_desc);
            float cum = 0.0f, cut = 0.0f;
            for (int i = 0; i < n; i++) { cum += sorted[i].val; if (cum >= s->topp) { cut = sorted[i].val; break; } }
            for (int i = 0; i < n; i++) if (s->probs[i] < cut) s->probs[i] = 0.0f;
            float sum = 0.0f;
            for (int i = 0; i < n; i++) sum += s->probs[i];
            if (sum > 0.0f) for (int i = 0; i < n; i++) s->probs[i] /= sum;
            free(sorted);
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += s->probs[i];
    unsigned int r = s->rng_state * 1664525u + 1013904223u;
    r ^= r >> 13; r *= 1274126177u; r ^= r >> 16;
    s->rng_state = r;
    float target = ((float)(r & 0xFFFFFF) / (float)0x1000000) * sum;
    float c = 0.0f;
    for (int i = 0; i < n; i++) { c += s->probs[i]; if (c >= target) return i; }
    return n - 1;
}

/* api.c owns the create/destroy wrappers so the public API symbols stay in one TU. */
