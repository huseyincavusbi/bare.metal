#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/context.h"
#include "baremetal/scheduler.h"
#include "backend/backend.h"
#include "kernels/registry.h"
#include "baremetal/graph.h"
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
    
    sess->sched = bmt_scheduler_create(ctx->backend_ctx, sess->reg, model->graph, model->arch.max_seq_len, model->kv_dim, model->arch.n_layers);
    bmt_scheduler_set_precision(sess->sched, model->precision);
    sess->logits = malloc(model->arch.vocab_size * sizeof(float));
    
    return sess;
}

void bm_destroy_session(bm_session_t* sess) {
    if (!sess) return;
    bmt_scheduler_destroy(sess->sched);
    bmk_registry_destroy(sess->reg);
    free(sess->logits);
    free(sess);
}

void bm_reset_session(bm_session_t* sess) {
    if (sess) sess->pos = 0;
}

float* bm_forward(bm_session_t* sess, const int* tokens, int n_tokens) {
    for (int i = 0; i < n_tokens; i++) {
        bm_step(sess, tokens[i]);
    }
    return sess->logits;
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
