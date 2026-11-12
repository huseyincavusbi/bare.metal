/* prefill_test.c -- batched prefill matches stepwise decode after the KV copy. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "utils/log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static double cosine(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int main(void) {
    const char* dir = "data/smollm2-135m";
    const int N = 64;
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "no backend\n"); return 1; }

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    if (!m) return 1;
    bm_load_weights(m, dir);
    if (!m->weight_buffer) {
        fprintf(stderr, "failed to load weights from %s\n", dir);
        free(m);
        bm_destroy(ctx);
        return 1;
    }
    m->precision = BM_PRECISION_FP32;
    bm_compile(m);
    bm_session_t* sess = bm_create_session(ctx, m);
    if (!sess) {
        fprintf(stderr, "session creation failed\n");
        bm_destroy_model(m);
        bm_destroy(ctx);
        return 1;
    }

    int V = m->arch.vocab_size;
    int* toks = malloc((size_t)N * sizeof(int));
    float* pre = malloc((size_t)V * sizeof(float));
    float* pre_next = malloc((size_t)V * sizeof(float));
    float* step = malloc((size_t)V * sizeof(float));
    float* step_next = malloc((size_t)V * sizeof(float));
    if (!toks || !pre || !pre_next || !step || !step_next) return 1;
    for (int i = 0; i < N; i++) toks[i] = (37 * i + 11) % V;
    int next = (37 * N + 11) % V;

    bm_reset_session(sess);
    float* logits = bm_forward(sess, toks, N);
    if (!logits) { fprintf(stderr, "prefill failed\n"); return 1; }
    memcpy(pre, logits, (size_t)V * sizeof(float));
    logits = bm_step(sess, next);
    if (!logits) { fprintf(stderr, "step after prefill failed\n"); return 1; }
    memcpy(pre_next, logits, (size_t)V * sizeof(float));

    bm_reset_session(sess);
    for (int i = 0; i < N; i++) {
        logits = bm_step(sess, toks[i]);
        if (!logits) { fprintf(stderr, "step %d failed\n", i); return 1; }
    }
    memcpy(step, logits, (size_t)V * sizeof(float));
    logits = bm_step(sess, next);
    if (!logits) { fprintf(stderr, "final step failed\n"); return 1; }
    memcpy(step_next, logits, (size_t)V * sizeof(float));

    double c_pre = cosine(pre, step, V);
    double c_next = cosine(pre_next, step_next, V);
    int ok = c_pre > 0.9999 && c_next > 0.9999
             && argmax(pre, V) == argmax(step, V)
             && argmax(pre_next, V) == argmax(step_next, V);
    printf("prefill cos %.7f argmax %d/%d\n", c_pre,
           argmax(pre, V), argmax(step, V));
    printf("kv step cos %.7f argmax %d/%d\n", c_next,
           argmax(pre_next, V), argmax(step_next, V));
    printf("\n=== prefill: %s ===\n", ok ? "PASS" : "FAIL");

    free(toks); free(pre); free(pre_next); free(step); free(step_next);
    bm_destroy_session(sess);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return ok ? 0 : 1;
}
