// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/context.h"
#include "baremetal/tokenizer.h"
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "kernels/registry.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

static backend_ctx_t* g_be;
#define B() do{enc=backend_encode_begin(g_be);}while(0)
#define D(kn,bufs,n,gx,gy,gz,tx,ty,tz) do{backend_encode_dispatch(enc,kn,bufs,NULL,n,gx,gy,gz,tx,ty,tz);}while(0)
#define C() do{backend_encode_commit(enc);backend_encode_wait(enc);}while(0)

int bm_run_tokens(bm_context_t* ctx, bm_model_t* m,
                  const int* prompt_ids, int n_prompt,
                  int steps, float temperature, int top_k, float top_p,
                  uint64_t seed,
                  bm_token_cb_t callback, void* user_data);

static void softmax_inplace(float* x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

typedef struct { float val; int idx; } float_idx_t;

static int cmp_float_desc(const void* a, const void* b) {
    float fa = ((const float_idx_t*)a)->val;
    float fb = ((const float_idx_t*)b)->val;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

static int sample_topk_topp(const float* logits, int n,
                            float temperature, int top_k, float top_p,
                            unsigned int rng_state) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float* probs = (float*)malloc(n * sizeof(float));
    if (!probs) return 0;
    for (int i = 0; i < n; i++) probs[i] = logits[i] / temperature;
    softmax_inplace(probs, n);

    if (top_k > 0 && top_k < n) {
        float_idx_t* sorted = (float_idx_t*)malloc(n * sizeof(float_idx_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(float_idx_t), cmp_float_desc);
            float thresh = sorted[top_k - 1].val;
            for (int i = 0; i < n; i++) if (probs[i] < thresh) probs[i] = 0.0f;
            free(sorted);
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += probs[i];
            if (s > 0.0f) for (int i = 0; i < n; i++) probs[i] /= s;
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        float_idx_t* sorted = (float_idx_t*)malloc(n * sizeof(float_idx_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(float_idx_t), cmp_float_desc);
            float cum = 0.0f; float cut = 0.0f;
            for (int i = 0; i < n; i++) { cum += sorted[i].val; if (cum >= top_p) { cut = sorted[i].val; break; } }
            for (int i = 0; i < n; i++) if (probs[i] < cut) probs[i] = 0.0f;
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += probs[i];
            if (s > 0.0f) for (int i = 0; i < n; i++) probs[i] /= s;
            free(sorted);
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probs[i];
    unsigned int r = (rng_state * 1664525u + 1013904223u);
    r ^= r >> 13; r *= 1274126177u; r ^= r >> 16;
    float target = ((float)(r & 0xFFFFFF) / (float)0x1000000) * sum;
    float c = 0.0f;
    int pick = 0;
    for (int i = 0; i < n; i++) { c += probs[i]; if (c >= target) { pick = i; break; } }
    free(probs);
    return pick;
}

int bm_run(bm_context_t* ctx, bm_model_t* m, const char* prompt,
           int steps, float temp, unsigned long long seed, const char* tok_path) {
    g_be=ctx->backend_ctx;
    fprintf(stderr,"Loading tokenizer...\n"); fflush(stderr);
    bm_tokenizer_t tok; bm_tokenizer_init(&tok, (char*)tok_path, m->arch.vocab_size);
    fprintf(stderr,"Tokenizer loaded, encoding...\n"); fflush(stderr);
    int* ptok = (int*)malloc(1024 * sizeof(int));
    int nt = 0;
    bm_tokenizer_encode(&tok, prompt, 1, 0, ptok, &nt);
    int rc = bm_run_tokens(ctx, m, ptok, nt, steps, temp, 0, 1.0f, seed, NULL, NULL);
    free(ptok);
    bm_tokenizer_free(&tok);
    return rc;
}

int bm_run_tokens(bm_context_t* ctx, bm_model_t* m,
                  const int* prompt_ids, int n_prompt,
                  int steps, float temperature, int top_k, float top_p,
                  uint64_t seed,
                  bm_token_cb_t callback, void* user_data) {
    if (n_prompt <= 0 || !prompt_ids) return -1;
    
    // 1. Compile model (Builds DAG and fuses ops)
    bm_compile(m);
    
    // 2. Create session (Initializes scheduler, allocates Metal buffers)
    bm_session_t* sess = bm_create_session(ctx, m);
    if (!sess) return -1;
    
    if (seed == 0) seed = (uint64_t)time(NULL);
    unsigned int rng_state = (unsigned int)seed;
    
    int next_token = -1;
    for (int pos = 0; pos < steps; pos++) {
        int token;
        if (pos < n_prompt) {
            token = prompt_ids[pos];
        } else {
            token = next_token;
        }
        
        // 3. Step forward
        float* logits = bm_step(sess, token);
        
        int V = m->arch.vocab_size;
        
        // 4. Debug output
        if (pos < n_prompt + 3) {
            float max_logit = -INFINITY, min_logit = INFINITY;
            for (int i = 0; i < V; i++) {
                if (logits[i] > max_logit) max_logit = logits[i];
                if (logits[i] < min_logit) min_logit = logits[i];
            }
            fprintf(stderr, " [logits: min=%.2f max=%.2f]", min_logit, max_logit);

            float_idx_t* sorted = malloc(V * sizeof(float_idx_t));
            for (int i = 0; i < V; i++) { sorted[i].val = logits[i]; sorted[i].idx = i; }
            qsort(sorted, V, sizeof(float_idx_t), cmp_float_desc);
            fprintf(stderr, " [top5:");
            for (int i = 0; i < 5; i++) fprintf(stderr, " %d(%.2f)", sorted[i].idx, sorted[i].val);
            fprintf(stderr, "]");
            free(sorted);
        }

        // 5. Sample next token
        if (pos >= n_prompt - 1) {
            next_token = sample_topk_topp(logits, V, temperature, top_k, top_p, rng_state++);
            if (callback) callback(next_token, user_data);
            if (pos < n_prompt + 5) {
                fprintf(stderr, " [gen token %d at pos %d]", next_token, pos);
            }
        }
        fprintf(stderr, "\r[step %d/%d]", pos + 1, steps);
        fflush(stderr);
    }
    printf("\n");
    
    bm_destroy_session(sess);
    return 0;
}
