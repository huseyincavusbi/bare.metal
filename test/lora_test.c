/* lora_test.c -- LoRA freezes the base and B=0 matches the base model. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/trainer.h"
#include "baremetal/lora.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static bm_trainer_t* make_trainer(bm_context_t* ctx, bm_model_t** out_model,
                                  int S, int rank, float lr, int targets) {
    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    if (!m) return NULL;
    bm_load_weights(m, "data/smollm2-135m");
    m->precision = bm_get_supported_precision(ctx);

    bm_train_config_t cfg = {0};
    cfg.learning_rate = lr;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.grad_accum_steps = 1;
    cfg.use_master_weights = 1;
    cfg.seq_len = S;
    cfg.lora_rank = rank;
    cfg.lora_alpha = 16.0f;
    cfg.lora_targets = targets;
    cfg.lora_seed = 1;

    bm_trainer_t* t = bmt_trainer_create(ctx, m, &cfg, S);
    if (!t) {
        bm_destroy_model(m);
        return NULL;
    }
    *out_model = m;
    return t;
}

int main(void) {
    const int S = 8;
    const int seq[9] = {5, 11, 23, 41, 17, 3, 29, 37, 7};

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "no context\n"); return 1; }

    float base_loss = 0.0f;
    {
        bm_model_t* m = NULL;
        bm_trainer_t* t = make_trainer(ctx, &m, S, 0, 0.0f, 0);
        if (!t) { fprintf(stderr, "base trainer failed\n"); bm_destroy(ctx); return 1; }
        base_loss = bm_train_step(t, seq, seq + 1, 1, S);
        printf("base loss: %.7f\n", base_loss);
        bm_destroy_trainer(t);
        bm_destroy_model(m);
    }

    bm_model_t* m = NULL;
    bm_trainer_t* t = make_trainer(ctx, &m, S, 8, 0.0f, BMT_LORA_Q);
    if (!t) { fprintf(stderr, "lora trainer failed\n"); bm_destroy(ctx); return 1; }

    check(m->n_lora == m->arch.n_layers, "q_proj adapter per layer");
    check(t->n_opt_states == 2 * m->n_lora, "optimizer state only for adapters");

    int trainable = 0, frozen = 0;
    bmt_graph_t* g = (bmt_graph_t*)m->graph;
    for (int i = 0; i < g->n_tensors; i++) {
        if (g->tensors[i].type != BMT_TENSOR_TYPE_WEIGHT) continue;
        if (g->tensors[i].trainable) trainable++;
        else frozen++;
    }
    check(trainable == 2 * m->n_lora, "exactly adapter weights are trainable");
    check(frozen > 0, "base weights are frozen");
    check(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_a) != NULL,
          "adapter A has a grad buffer");
    check(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_b) != NULL,
          "adapter B has a grad buffer");

    float lora_loss = bm_train_step(t, seq, seq + 1, 1, S);
    printf("lora loss: %.7f (base %.7f)\n", lora_loss, base_loss);
    check(fabsf(lora_loss - base_loss) < 1e-5f, "B=0 matches the base loss");

    {
        float* ga = backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_a));
        float* gb = backend_buffer_map(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_b));
        float amax = 0.0f, bmax = 0.0f;
        for (size_t i = 0; i < (size_t)m->lora[0].r * m->lora[0].c; i++)
            if (fabsf(ga[i]) > amax) amax = fabsf(ga[i]);
        for (size_t i = 0; i < (size_t)m->lora[0].oc * m->lora[0].r; i++)
            if (fabsf(gb[i]) > bmax) bmax = fabsf(gb[i]);
        backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_a));
        backend_buffer_unmap(bmt_scheduler_get_grad_buffer(t->sched, m->lora[0].t_b));
        check(amax == 0.0f, "A gradient is zero at B=0");
        check(bmax > 0.0f, "B gradient is non-zero at B=0");
    }

    size_t before = (size_t)m->n_parameters * sizeof(float);
    float* base_before = malloc(before);
    memcpy(base_before, m->weight_buffer, before);

    t->cfg.learning_rate = 1e-3f;
    float l0 = bm_train_step(t, seq, seq + 1, 1, S);
    float l1 = bm_train_step(t, seq + 1, seq + 2, 1, S);
    check(isfinite(l0) && isfinite(l1), "LoRA steps produce finite loss");
    check(memcmp(base_before, m->weight_buffer, before) == 0,
          "base host weights are untouched");

    int changed = 0;
    for (int i = 0; i < t->n_opt_states; i++) {
        bmt_adamw_state_t* st = &t->opt_states[i];
        if (!st->master_w) continue;
        float* master = backend_buffer_map(st->master_w);
        const float* init = st->weight_ptr;
        for (size_t j = 0; j < st->n_params; j++) {
            if (master[j] != init[j]) { changed++; break; }
        }
        backend_buffer_unmap(st->master_w);
    }
    check(changed > 0, "adapter master weights update");

    free(base_before);
    bm_destroy_trainer(t);
    bm_destroy_model(m);

    {
        bm_model_t* m2 = NULL;
        bm_trainer_t* t2 = make_trainer(ctx, &m2, S, 8, 0.0f, BMT_LORA_ALL);
        if (!t2) {
            check(0, "all-target LoRA trainer");
        } else {
            int expected = m2->arch.n_layers * (m2->arch.gated_mlp ? 7 : 6);
            check(m2->n_lora == expected, "all-target adapter count");
            float loss = bm_train_step(t2, seq, seq + 1, 1, S);
            check(fabsf(loss - base_loss) < 1e-5f, "all targets B=0 matches base");
            bm_destroy_trainer(t2);
            bm_destroy_model(m2);
        }
    }

    bm_destroy(ctx);

    printf("\n=== lora: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
