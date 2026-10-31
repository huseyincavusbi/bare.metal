// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/context.h"
#include "baremetal/scheduler.h"
#include "backend/backend.h"
#include "kernels/registry.h"
#include "baremetal/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct bm_session_s {
    bm_context_t* ctx;
    bm_model_t* model;
    bmt_scheduler_t* sched;
    bmk_registry_t* reg;

    int pos;
    float* logits;

    bmt_scheduler_t* prefill_sched;
    bmt_graph_t*     prefill_graph;
    int              prefill_S;
    int*             prefill_k;
    int*             prefill_v;
};

bm_session_t* bm_create_session(bm_context_t* ctx, bm_model_t* model) {
    bm_session_t* sess = calloc(1, sizeof(bm_session_t));
    sess->ctx = ctx;
    sess->model = model;
    sess->pos = 0;
    
    sess->reg = bmk_registry_create(ctx->backend_ctx);
    bmk_register(sess->reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(sess->reg, BMK_OP_MATMUL, BMK_VARIANT_TILED, "matmul_forward_tiled");
    bmk_register(sess->reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(sess->reg, BMK_OP_NORM_RMS, BMK_VARIANT_TILED, "rmsnorm_forward_v2");
    bmk_register(sess->reg, BMK_OP_NORM_LAYER, BMK_VARIANT_NAIVE, "layernorm_forward");
    bmk_register(sess->reg, BMK_OP_NORM_LAYER, BMK_VARIANT_TILED, "layernorm_forward_v2");
    bmk_register(sess->reg, BMK_OP_ACT_GELU, BMK_VARIANT_NAIVE, "gelu_forward");
    bmk_register(sess->reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");
    bmk_register(sess->reg, BMK_OP_POS_ENC_ROPE, BMK_VARIANT_NAIVE, "rope_forward");
    bmk_register(sess->reg, BMK_OP_ATTENTION, BMK_VARIANT_NAIVE, "attention_forward");
    bmk_register(sess->reg, BMK_OP_ATTENTION, BMK_VARIANT_FLASH, "attention_forward_flash");
    bmk_register(sess->reg, BMK_OP_FUSED_RESIDUAL_NORM, BMK_VARIANT_NAIVE, "residual_rmsnorm_forward");
    bmk_register(sess->reg, BMK_OP_FUSED_CLASSIFIER, BMK_VARIANT_NAIVE, "rmsnorm_matmul_forward");
    
    sess->sched = bmt_scheduler_create_with_precision(ctx->backend_ctx, sess->reg, model->graph, model->arch.max_seq_len, model->kv_dim, model->arch.n_layers, model->precision, 0);
    bmt_model_release_host_weights(model);
    sess->logits = malloc(model->arch.vocab_size * sizeof(float));
    
    return sess;
}

void bm_destroy_session(bm_session_t* sess) {
    if (!sess) return;
    if (sess->prefill_sched) bmt_scheduler_destroy(sess->prefill_sched);
    if (sess->prefill_graph) bmt_graph_destroy(sess->prefill_graph);
    free(sess->prefill_k);
    free(sess->prefill_v);
    bmt_scheduler_destroy(sess->sched);
    bmk_registry_destroy(sess->reg);
    free(sess->logits);
    free(sess);
}

void bm_profile_begin(bm_session_t* sess) {
    if (sess) bmt_scheduler_profile_begin(sess->sched);
}

void bm_profile_end(bm_session_t* sess, double* attention_ms, double* mlp_ms, double* other_ms) {
    if (sess) bmt_scheduler_profile_end(sess->sched, attention_ms, mlp_ms, other_ms);
}

void bm_reset_session(bm_session_t* sess) {
    if (sess) sess->pos = 0;
}

static float* bm_prefill(bm_session_t* sess, const int* tokens, int n_tokens) {
    bm_model_t* m = sess->model;
    int D = m->arch.dim;
    int L = m->arch.n_layers;
    int V = m->arch.vocab_size;
    int MS = m->arch.max_seq_len;

    if (sess->prefill_S != n_tokens) {
        if (sess->prefill_sched) bmt_scheduler_destroy(sess->prefill_sched);
        if (sess->prefill_graph) bmt_graph_destroy(sess->prefill_graph);
        free(sess->prefill_k);
        free(sess->prefill_v);

        sess->prefill_k = malloc((size_t)L * sizeof(int));
        sess->prefill_v = malloc((size_t)L * sizeof(int));
        sess->prefill_graph = bmt_graph_build_prefill(m, n_tokens, sess->prefill_k, sess->prefill_v);
        sess->prefill_sched = bmt_scheduler_create_shared(
            sess->ctx->backend_ctx, sess->reg, sess->prefill_graph, MS,
            m->kv_dim, L, m->precision, 0, sess->sched);
        bmt_scheduler_set_prefill(sess->prefill_sched, 1);
        sess->prefill_S = n_tokens;
    }

    float* x = malloc((size_t)n_tokens * D * sizeof(float));
    for (int i = 0; i < n_tokens; i++) {
        memcpy(x + (size_t)i * D,
               m->token_embedding_table + (size_t)tokens[i] * D,
               (size_t)D * sizeof(float));
        if (m->arch.embed_scale) {
            float esc = sqrtf((float)D);
            for (int d = 0; d < D; d++) x[(size_t)i * D + d] *= esc;
        }
        if (m->arch.pos_enc == BM_POS_LEARNED && m->wpe) {
            for (int d = 0; d < D; d++) x[(size_t)i * D + d] += m->wpe[(size_t)i * D + d];
        }
    }

    bmt_scheduler_set_input(sess->prefill_sched, 0, x, (size_t)n_tokens * D * sizeof(float));
    bmt_scheduler_forward_train(sess->prefill_sched, n_tokens);
    bmt_scheduler_get_output(sess->prefill_sched,
                             sess->prefill_graph->n_tensors - 1,
                             sess->logits, (size_t)V * sizeof(float));
    bmt_scheduler_copy_prefill_kv(sess->sched, sess->prefill_sched,
                                  n_tokens, sess->prefill_k, sess->prefill_v);

    sess->pos = n_tokens;
    free(x);
    return sess->logits;
}

float* bm_forward(bm_session_t* sess, const int* tokens, int n_tokens) {
    if (n_tokens <= 0) return sess->logits;
    /* The q4/q8 matmul kernels are decode-only (batch 1); prefill falls back
     * to the token loop until batched quantized matmuls exist. */
    if (n_tokens == 1 || sess->pos != 0 || sess->model->quant_bits != 0 ||
        n_tokens > sess->model->arch.max_seq_len) {
        for (int i = 0; i < n_tokens; i++) bm_step(sess, tokens[i]);
        return sess->logits;
    }
    return bm_prefill(sess, tokens, n_tokens);
}

float* bm_step(bm_session_t* sess, int token) {
    int D = sess->model->arch.dim;
    float* wte = sess->model->token_embedding_table;
    float* x_in = malloc(D * sizeof(float));
    memcpy(x_in, wte + token * D, D * sizeof(float));
    
    if (sess->model->arch.embed_scale) {
        float esc = sqrtf((float)D);
        for(int i=0; i<D; i++) x_in[i] *= esc;
    }
    
    if (sess->model->arch.pos_enc == BM_POS_LEARNED && sess->model->wpe) {
        for(int i=0; i<D; i++) x_in[i] += sess->model->wpe[sess->pos * D + i];
    }
    
    int t_x_id = 0;
    bmt_scheduler_set_input(sess->sched, t_x_id, x_in, D * sizeof(float));
    
    bmt_scheduler_run(sess->sched, sess->pos, 1);
    
    int V = sess->model->arch.vocab_size;
    bmt_graph_t* g = (bmt_graph_t*)sess->model->graph;
    int t_logits_id = g->n_tensors - 1;
    bmt_scheduler_get_output(sess->sched, t_logits_id, sess->logits, V * sizeof(float));
    
    sess->pos++;
    free(x_in);
    
    return sess->logits;
}
