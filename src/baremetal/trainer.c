#include "baremetal/trainer.h"
#include "baremetal/context.h"
#include "baremetal/graph.h"
#include "baremetal/compiler.h"
#include "baremetal/model.h"
#include "baremetal/types.h"
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

    /* build the training graph (separate residuals, S-sized) */
    bmt_graph_build_train(model, S);
    /* TODO: fusion requires in-place residuals, but training uses separate
     * residual tensors for backprop. Disable fusion for now. */
    /* bmt_compiler_run(model->graph); */
    bmt_graph_t* g = (bmt_graph_t*)model->graph;
    t->t_x_id = 0;
    t->t_logits_id = g->n_tensors - 1;

    /* registry: forward kernels (training uses naive variants) */
    t->reg = bmk_registry_create(ctx->backend_ctx);
    bmk_register(t->reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(t->reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(t->reg, BMK_OP_FUSED_RESIDUAL_NORM, BMK_VARIANT_NAIVE, "residual_rmsnorm_forward");
    bmk_register(t->reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");

    t->sched = bmt_scheduler_create_with_precision(ctx->backend_ctx, t->reg, g, S, model->kv_dim, model->arch.n_layers, model->precision);
    t->precision = model->precision;

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
        /* For mixed precision: keep fp32 master copy for AdamW (2D weights only) */
        if (t->precision != BM_PRECISION_FP32 && wt->n_dims >= 2) {
            st->master_w = backend_buffer_alloc(ctx->backend_ctx, n * sizeof(float));
            memcpy(backend_buffer_map(st->master_w), wt->weight_ptr, n * sizeof(float));
            backend_buffer_unmap(st->master_w);
        } else {
            st->master_w = NULL;
        }
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
        if (t->opt_states[i].master_w) backend_buffer_free(t->opt_states[i].master_w);
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

void bm_save_state(bm_trainer_t* t, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        BMT_LOG_ERROR("bm_save_state: cannot open %s", path);
        return;
    }

    int step = t->step;
    int n_states = t->n_opt_states;
    fwrite(&step, sizeof(int), 1, f);
    fwrite(&n_states, sizeof(int), 1, f);

    for (int i = 0; i < n_states; i++) {
        bmt_adamw_state_t* st = &t->opt_states[i];
        size_t n = st->n_params;

        float* w_cpu = malloc(n * sizeof(float));
        float* m_cpu = malloc(n * sizeof(float));
        float* v_cpu = malloc(n * sizeof(float));

        float* w_gpu = st->master_w
            ? (float*)backend_buffer_map(st->master_w)
            : (float*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
        float* m_gpu = backend_buffer_map(st->m);
        float* v_gpu = backend_buffer_map(st->v);

        memcpy(w_cpu, w_gpu, n * sizeof(float));
        memcpy(m_cpu, m_gpu, n * sizeof(float));
        memcpy(v_cpu, v_gpu, n * sizeof(float));

        if (st->master_w) backend_buffer_unmap(st->master_w);
        else backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
        backend_buffer_unmap(st->m);
        backend_buffer_unmap(st->v);

        fwrite(&n, sizeof(size_t), 1, f);
        fwrite(w_cpu, sizeof(float), n, f);
        fwrite(m_cpu, sizeof(float), n, f);
        fwrite(v_cpu, sizeof(float), n, f);

        free(w_cpu);
        free(m_cpu);
        free(v_cpu);
    }

    fclose(f);
    BMT_LOG_INFO("bm_save_state: saved step=%d, %d weight tensors to %s", step, n_states, path);
}

void bm_load_state(bm_trainer_t* t, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        BMT_LOG_ERROR("bm_load_state: cannot open %s", path);
        return;
    }

    int step, n_states;
    if (fread(&step, sizeof(int), 1, f) != 1 ||
        fread(&n_states, sizeof(int), 1, f) != 1) {
        BMT_LOG_ERROR("bm_load_state: failed to read header");
        fclose(f);
        return;
    }

    if (n_states != t->n_opt_states) {
        BMT_LOG_ERROR("bm_load_state: state has %d tensors, trainer has %d", n_states, t->n_opt_states);
        fclose(f);
        return;
    }

    t->step = step;

    for (int i = 0; i < n_states; i++) {
        bmt_adamw_state_t* st = &t->opt_states[i];
        size_t n;
        if (fread(&n, sizeof(size_t), 1, f) != 1 || n != st->n_params) {
            BMT_LOG_ERROR("bm_load_state: tensor %d size mismatch", i);
            fclose(f);
            return;
        }

        float* w_cpu = malloc(n * sizeof(float));
        float* m_cpu = malloc(n * sizeof(float));
        float* v_cpu = malloc(n * sizeof(float));

        if (fread(w_cpu, sizeof(float), n, f) != n ||
            fread(m_cpu, sizeof(float), n, f) != n ||
            fread(v_cpu, sizeof(float), n, f) != n) {
            BMT_LOG_ERROR("bm_load_state: failed to read tensor %d data", i);
            free(w_cpu);
            free(m_cpu);
            free(v_cpu);
            fclose(f);
            return;
        }

        float* w_gpu = st->master_w
            ? (float*)backend_buffer_map(st->master_w)
            : (float*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
        float* m_gpu = backend_buffer_map(st->m);
        float* v_gpu = backend_buffer_map(st->v);

        memcpy(w_gpu, w_cpu, n * sizeof(float));
        memcpy(m_gpu, m_cpu, n * sizeof(float));
        memcpy(v_gpu, v_cpu, n * sizeof(float));

        if (st->master_w) backend_buffer_unmap(st->master_w);
        else backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
        backend_buffer_unmap(st->m);
        backend_buffer_unmap(st->v);

        free(w_cpu);
        free(m_cpu);
        free(v_cpu);
    }

    fclose(f);
    BMT_LOG_INFO("bm_load_state: loaded step=%d, %d weight tensors from %s", step, n_states, path);
}

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
        backend_buffer_t* b_g = bmt_scheduler_get_grad_buffer(t->sched, wid);
        int n = (int)st->n_params;
        /* In mixed precision: AdamW updates fp32 master, then we convert to bf16 working */
        backend_buffer_t* b_w = st->master_w ? st->master_w : bmt_scheduler_get_buffer(t->sched, wid);
        backend_buffer_t* bufs[] = { b_w, b_g, st->m, st->v, st->par, t->k_adamw_hp };
        backend_encode_dispatch(enc, t->k_adamw, bufs, NULL, 6, n, 1, 1, n < 256 ? n : 256, 1, 1);
    }
    backend_encode_commit(enc); backend_encode_wait(enc);

    /* Mixed precision: convert fp32 master → bf16/fp16 working buffer.
     * Only convert 2D weights (norm weights stay fp32 on GPU). */
    if (t->precision != BM_PRECISION_FP32) {
        bmt_graph_t* g = (bmt_graph_t*)t->model->graph;
        for (int i = 0; i < t->n_opt_states; i++) {
            bmt_adamw_state_t* st = &t->opt_states[i];
            int wid = st->grad_tensor_id;
            if (!st->master_w) continue;
            int n = (int)st->n_params;
            float* master = (float*)backend_buffer_map(st->master_w);
            if (t->precision == BM_PRECISION_BF16) {
                bm_bf16_t* work = (bm_bf16_t*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, wid));
                bm_f32_to_bf16_array(master, work, n);
                backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, wid));
            } else if (t->precision == BM_PRECISION_FP16) {
                bm_fp16_t* work = (bm_fp16_t*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, wid));
                for (int j = 0; j < n; j++) {
                    __fp16 h = (__fp16)master[j];
                    memcpy(&work[j], &h, sizeof(bm_fp16_t));
                }
                backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, wid));
            }
            backend_buffer_unmap(st->master_w);
        }
    }
}

float bm_train_step(bm_trainer_t* t, const int* inputs, const int* targets, int B, int T) {
    int S = T;
    if (S != t->S) {
        BMT_LOG_ERROR("bm_train_step: S=%d != trainer S=%d (rebuild trainer for new S)", S, t->S);
        return -1.0f;
    }

    /* zero all grad buffers once at the start (gradients accumulate across B sequences) */
    bmt_graph_t* g = (bmt_graph_t*)t->model->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        size_t n = 1;
        for (int d = 0; d < g->tensors[i].n_dims; d++) n *= g->tensors[i].dims[d];
        if (n == 0) n = 1024;
        memset(backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, i)), 0, n * sizeof(float));
        backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, i));
    }

    /* Sequential batching: process B sequences one at a time, accumulating gradients */
    float total_loss = 0.0f;
    for (int b = 0; b < B; b++) {
        const int* seq_inputs = inputs + b * S;
        const int* seq_targets = targets + b * S;

        /* 1. embedding lookup + forward */
        embed_lookup(t, seq_inputs, S);
        bmt_scheduler_forward_train(t->sched, S);

        /* 2. loss seed (xent_backward fills grad_logits) */
        float loss = bmt_scheduler_xent_backward(t->sched, t->t_logits_id, seq_targets, S, t->model->arch.vocab_size);
        total_loss += loss;

        /* 3. backward walk (gradients accumulate via atomic adds) */
        bmt_scheduler_backward(t->sched);

        /* 4. embedding backward (scatter grad_x into grad_wte + tied sum) */
        embed_backward(t, seq_inputs, S);
    }

    /* 5. AdamW step (once per batch, after all B sequences) */
    adamw_apply(t);

    return total_loss / (float)B;
}

#endif /* BAREMETAL_TRAIN */
