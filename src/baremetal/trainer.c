#include "baremetal/trainer.h"
#include "baremetal/context.h"
#include "baremetal/graph.h"
#include "baremetal/model.h"
#include "kernels/registry.h"
#include "backend/backend.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef BAREMETAL_TRAIN

static backend_buffer_t* alloc_zero(backend_ctx_t* be, size_t bytes) {
    backend_buffer_t* b = backend_buffer_alloc(be, bytes);
    memset(backend_buffer_map(b), 0, bytes);
    backend_buffer_unmap(b);
    return b;
}

bm_trainer_t* bmt_trainer_create(bm_context_t* ctx, bm_model_t* model,
                                 const bm_train_config_t* cfg, int S) {
    bm_trainer_t* t = calloc(1, sizeof(bm_trainer_t));
    t->ctx = ctx;
    t->model = model;
    t->cfg = *cfg;
    t->step = 0;
    t->S = S;

    /* build the training graph (separate residuals, no fusion, S-sized) */
    bmt_graph_build_train(model, S);
    bmt_graph_t* g = (bmt_graph_t*)model->graph;
    t->t_x_id = 0;
    t->t_logits_id = g->n_tensors - 1;

    /* registry: forward kernels (training uses naive variants) */
    t->reg = bmk_registry_create(ctx->backend_ctx);
    bmk_register(t->reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(t->reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(t->reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");

    t->sched = bmt_scheduler_create(ctx->backend_ctx, t->reg, g, S, model->kv_dim, model->arch.n_layers);

    /* AdamW state: one entry per WEIGHT tensor in the graph. */
    t->n_opt_states = 0;
    int cap = 64;
    t->opt_states = calloc(cap, sizeof(bmt_adamw_state_t));
    for (int i = 0; i < g->n_tensors; i++) {
        bmt_tensor_t* wt = &g->tensors[i];
        if (wt->type != BMT_TENSOR_TYPE_WEIGHT || !wt->weight_ptr) continue;
        size_t n = 1;
        for (int d = 0; d < wt->n_dims; d++) n *= wt->dims[d];
        if (n == 0) continue;
        if (t->n_opt_states >= cap) { cap *= 2; t->opt_states = realloc(t->opt_states, cap*sizeof(bmt_adamw_state_t)); }
        bmt_adamw_state_t* st = &t->opt_states[t->n_opt_states++];
        st->weight_ptr = wt->weight_ptr;
        st->n_params = n;
        st->grad_tensor_id = i;
        st->m = alloc_zero(ctx->backend_ctx, n * sizeof(float));
        st->v = alloc_zero(ctx->backend_ctx, n * sizeof(float));
        st->par = backend_buffer_alloc(ctx->backend_ctx, 16);
        int* pp = backend_buffer_map(st->par);
        pp[0] = (int)n;
        backend_buffer_unmap(st->par);
    }

    t->k_adamw = backend_kernel_create(ctx->backend_ctx, "adamw_step");
    t->k_adamw_hp = backend_buffer_alloc(ctx->backend_ctx, 7 * sizeof(float));
    t->k_embed_bwd = backend_kernel_create(ctx->backend_ctx, "embedding_backward");
    t->k_embed_param = backend_buffer_alloc(ctx->backend_ctx, 16 * sizeof(int));

    int D = model->arch.dim;
    t->x_in = malloc((size_t)S * D * sizeof(float));

    BMT_LOG_INFO("Trainer created: %d weight tensors, S=%d, D=%d", t->n_opt_states, S, D);
    return t;
}

void bm_destroy_trainer(bm_trainer_t* t) {
    if (!t) return;
    for (int i = 0; i < t->n_opt_states; i++) {
        backend_buffer_free(t->opt_states[i].m);
        backend_buffer_free(t->opt_states[i].v);
        backend_buffer_free(t->opt_states[i].par);
    }
    free(t->opt_states);
    if (t->k_adamw) backend_kernel_destroy(t->k_adamw);
    if (t->k_embed_bwd) backend_kernel_destroy(t->k_embed_bwd);
    if (t->k_adamw_hp) backend_buffer_free(t->k_adamw_hp);
    if (t->k_embed_param) backend_buffer_free(t->k_embed_param);
    bmt_scheduler_destroy(t->sched);
    bmk_registry_destroy(t->reg);
    free(t->x_in);
    free(t);
}

void bm_load_state(bm_trainer_t* t, const char* path) { (void)t; (void)path; }
void bm_save_state(bm_trainer_t* t, const char* path) { (void)t; (void)path; }

/* find the graph weight tensor id for a given weight_ptr */
static int find_wt_id(bmt_graph_t* g, void* ptr) {
    for (int i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].type == BMT_TENSOR_TYPE_WEIGHT && g->tensors[i].weight_ptr == ptr) return i;
    return -1;
}

/* embedding lookup: x[s,:] = wte[token[s]*D + :]  (SmolLM2: no scale, no wpe) */
static void embed_lookup(bm_trainer_t* t, const int* tokens, int S) {
    int D = t->model->arch.dim;
    float* wte = t->model->token_embedding_table;
    for (int s = 0; s < S; s++)
        memcpy(t->x_in + s*D, wte + (size_t)tokens[s]*D, D*sizeof(float));
    bmt_scheduler_set_input(t->sched, t->t_x_id, t->x_in, (size_t)S*D*sizeof(float));
}

/* embedding backward: scatter-add grad_x into grad_wte. Also add grad_wcls
 * into grad_wte when weights are tied (wcls == wte). */
static void embed_backward(bm_trainer_t* t, const int* tokens, int S) {
    bmt_graph_t* g = (bmt_graph_t*)t->model->graph;
    int D = t->model->arch.dim;
    backend_ctx_t* be = t->ctx->backend_ctx;

    /* grad_x lives in grad_buffers[t_x_id]; scatter-add into grad_wte. */
    int wte_id = find_wt_id(g, t->model->token_embedding_table);
    int* p = backend_buffer_map(t->k_embed_param);
    p[0] = D;
    backend_buffer_unmap(t->k_embed_param);

    /* need a tokens buffer */
    backend_buffer_t* b_tok = backend_buffer_alloc(be, S * sizeof(int));
    memcpy(backend_buffer_map(b_tok), tokens, S * sizeof(int));
    backend_buffer_unmap(b_tok);

    /* grad_wte is already zeroed at the start of bm_train_step (all grad buffers).
     * For tied weights, the classifier's matmul_backward_w already wrote its grad
     * into the same buffer (wcls==wte). DO NOT zero here — the atomic scatter
     * accumulates on top of the classifier contribution. */
    size_t VD = (size_t)t->model->arch.vocab_size * D;
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = { b_tok, bmt_scheduler_get_grad_buffer(t->sched, t->t_x_id),
                                     bmt_scheduler_get_grad_buffer(t->sched, wte_id), t->k_embed_param };
        backend_encode_dispatch(enc, t->k_embed_bwd, bufs, NULL, 4, S*D, 1, 1, S*D, 1, 1);
        backend_encode_commit(enc); backend_encode_wait(enc);
    }
    backend_buffer_free(b_tok);

    /* tied weights: grad_wte += grad_wcls (the classifier weight grad). */
    if (t->model->arch.weight_tie && t->model->wcls == t->model->token_embedding_table) {
        int wcls_id = find_wt_id(g, t->model->wcls);
        if (wcls_id >= 0 && wcls_id != wte_id) {
            float* gwte = backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, wte_id));
            float* gwcls = backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, wcls_id));
            for (size_t i = 0; i < VD; i++) gwte[i] += gwcls[i];
            backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, wte_id));
            backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, wcls_id));
        }
    }
}

/* apply AdamW to every weight using its optimizer state + the graph grad buffer. */
static void adamw_apply(bm_trainer_t* t) {
    t->step++;
    float b1 = t->cfg.beta1, b2 = t->cfg.beta2;
    float bias1 = 1.0f - powf(b1, (float)t->step);
    float bias2 = 1.0f - powf(b2, (float)t->step);
    float hp[7] = { t->cfg.learning_rate, b1, b2, t->cfg.epsilon, t->cfg.weight_decay, bias1, bias2 };
    memcpy(backend_buffer_map(t->k_adamw_hp), hp, sizeof(hp));
    backend_buffer_unmap(t->k_adamw_hp);

    backend_encoder_t* enc = backend_encode_begin(t->ctx->backend_ctx);
    for (int i = 0; i < t->n_opt_states; i++) {
        bmt_adamw_state_t* st = &t->opt_states[i];
        int wid = st->grad_tensor_id;
        backend_buffer_t* b_w = bmt_scheduler_get_buffer(t->sched, wid);       /* fp32 master weight */
        backend_buffer_t* b_g = bmt_scheduler_get_grad_buffer(t->sched, wid);  /* gradient */
        int n = (int)st->n_params;
        backend_buffer_t* bufs[] = { b_w, b_g, st->m, st->v, st->par, t->k_adamw_hp };
        backend_encode_dispatch(enc, t->k_adamw, bufs, NULL, 6, n, 1, 1, n < 256 ? n : 256, 1, 1);
    }
    backend_encode_commit(enc); backend_encode_wait(enc);

    /* Weights stay on the GPU (the forward reads from GPU buffers, which AdamW
     * updated in-place). No per-step copy-back to CPU — that was 272 host syncs.
     * Grad buffers are already zeroed at the start of the next bm_train_step. */
}

float bm_train_step(bm_trainer_t* t, const int* inputs, const int* targets, int B, int T) {
    (void)B;   /* B=1 for now (single sequence of length T) */
    int S = T;
    if (S != t->S) {
        BMT_LOG_ERROR("bm_train_step: S=%d != trainer S=%d (rebuild trainer for new S)", S, t->S);
        return -1.0f;
    }

    /* zero all grad buffers (the walk accumulates into them) */
    bmt_graph_t* g = (bmt_graph_t*)t->model->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        size_t n = 1;
        for (int d = 0; d < g->tensors[i].n_dims; d++) n *= g->tensors[i].dims[d];
        if (n == 0) n = 1024;
        memset(backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, i)), 0, n * sizeof(float));
        backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, i));
    }

    /* 1. embedding lookup + forward */
    embed_lookup(t, inputs, S);
    bmt_scheduler_forward_train(t->sched, S);

    /* 2. loss seed (xent_backward fills grad_logits) */
    float loss = bmt_scheduler_xent_backward(t->sched, t->t_logits_id, targets, S, t->model->arch.vocab_size);

    /* 3. backward walk */
    bmt_scheduler_backward(t->sched);

    /* 4. embedding backward (scatter grad_x into grad_wte + tied sum) */
    embed_backward(t, inputs, S);

    /* 5. AdamW step */
    adamw_apply(t);

    return loss;
}

#endif /* BAREMETAL_TRAIN */
