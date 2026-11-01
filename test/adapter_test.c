/* adapter_test.c -- save a trained LoRA adapter and merge it into a fresh base. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/trainer.h"
#include "baremetal/lora.h"
#include "baremetal/scheduler.h"
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

static bmt_adamw_state_t* find_state(bm_trainer_t* t, int id) {
    for (int i = 0; i < t->n_opt_states; i++)
        if (t->opt_states[i].grad_tensor_id == id) return &t->opt_states[i];
    return NULL;
}

static float* state_ptr(bm_trainer_t* t, bmt_adamw_state_t* st, int* mapped) {
    *mapped = st->master_w != NULL;
    if (st->master_w) return (float*)backend_buffer_map(st->master_w);
    return (float*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
}

static void state_unmap(bm_trainer_t* t, bmt_adamw_state_t* st, int mapped) {
    if (mapped) backend_buffer_unmap(st->master_w);
    else backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
}

int main(void) {
    const int S = 8;
    const int seq[9] = {5, 11, 23, 41, 17, 3, 29, 37, 7};
    const char* path = "/tmp/bm_adapter_test.bin";

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) return 1;

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, "data/smollm2-135m");
    m->precision = BM_PRECISION_FP32;

    bm_train_config_t cfg = {0};
    cfg.learning_rate = 1e-3f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.grad_accum_steps = 1;
    cfg.use_master_weights = 1;
    cfg.lora_rank = 8;
    cfg.lora_alpha = 16.0f;
    cfg.lora_targets = BMT_LORA_Q;
    cfg.lora_seed = 1;

    bm_trainer_t* t = bmt_trainer_create(ctx, m, &cfg, S);
    if (!t) { fprintf(stderr, "trainer failed\n"); return 1; }
    bm_train_step(t, seq, seq + 1, 1, S);
    bm_train_step(t, seq + 1, seq + 2, 1, S);

    check(bm_save_adapter(t, path) == 0, "save adapter");

    bmt_lora_adapter_t* a = &m->lora[0];
    bmt_adamw_state_t* sa = find_state(t, a->t_a);
    bmt_adamw_state_t* sb = find_state(t, a->t_b);
    int ma = 0, mb = 0;
    float* av = state_ptr(t, sa, &ma);
    float* bv = state_ptr(t, sb, &mb);
    int oc = a->oc, c = a->c, r = a->r;
    float* expected = malloc((size_t)oc * c * sizeof(float));
    memcpy(expected, m->qw, (size_t)oc * c * sizeof(float));
    for (int row = 0; row < oc; row++) {
        for (int col = 0; col < c; col++) {
            float sum = 0.0f;
            for (int k = 0; k < r; k++)
                sum += bv[(size_t)row * r + k] * av[(size_t)k * c + col];
            expected[(size_t)row * c + col] += a->scale * sum;
        }
    }
    state_unmap(t, sa, ma);
    state_unmap(t, sb, mb);
    bm_destroy_trainer(t);
    bm_destroy_model(m);

    bm_model_t* m2 = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m2, "data/smollm2-135m");
    m2->precision = BM_PRECISION_FP32;
    check(bm_load_adapter(m2, path) == 0, "load and merge adapter");

    float maxabs = 0.0f;
    for (size_t i = 0; i < (size_t)oc * c; i++) {
        float d = fabsf(m2->qw[i] - expected[i]);
        if (d > maxabs) maxabs = d;
    }
    printf("merge maxabs: %.3g\n", maxabs);
    check(maxabs < 1e-6f, "merged weights match B*A");

    free(expected);
    bm_destroy_model(m2);
    bm_destroy(ctx);
    remove(path);

    printf("\n=== adapter: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
