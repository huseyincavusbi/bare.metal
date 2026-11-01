/* grad_accum_test.c -- weight-gradient accumulation across micro-steps.
 *
 * 1. Kernel level: matmul_backward_w and _tiled must add into gw, not
 *    overwrite it.
 * 2. Trainer level: B=2/G=1 must produce the same weights as B=1/G=2 with the
 *    same two sequences, because both windows average over two micro-steps.
 */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/trainer.h"
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void upload(backend_buffer_t* buf, const void* src, size_t bytes) {
    memcpy(backend_buffer_map(buf), src, bytes);
    backend_buffer_unmap(buf);
}

static void download(backend_buffer_t* buf, void* dst, size_t bytes) {
    memcpy(dst, backend_buffer_map(buf), bytes);
    backend_buffer_unmap(buf);
}

static void zero(backend_buffer_t* buf, size_t bytes) {
    memset(backend_buffer_map(buf), 0, bytes);
    backend_buffer_unmap(buf);
}

static int check_dw_kernel(backend_ctx_t* be, const char* name, int tiled) {
    const int BT = 4, C = 8, OC = 8;
    size_t nw = (size_t)OC * C;

    float* gout1 = malloc(BT * OC * sizeof(float));
    float* gout2 = malloc(BT * OC * sizeof(float));
    float* inp1  = malloc(BT * C * sizeof(float));
    float* inp2  = malloc(BT * C * sizeof(float));
    for (int i = 0; i < BT * OC; i++) {
        gout1[i] = sinf(0.13f * (float)i);
        gout2[i] = cosf(0.07f * (float)i);
    }
    for (int i = 0; i < BT * C; i++) {
        inp1[i] = cosf(0.11f * (float)i);
        inp2[i] = sinf(0.05f * (float)i);
    }

    backend_buffer_t* b_g1 = backend_buffer_alloc(be, BT * OC * sizeof(float));
    backend_buffer_t* b_g2 = backend_buffer_alloc(be, BT * OC * sizeof(float));
    backend_buffer_t* b_i1 = backend_buffer_alloc(be, BT * C * sizeof(float));
    backend_buffer_t* b_i2 = backend_buffer_alloc(be, BT * C * sizeof(float));
    backend_buffer_t* b_w  = backend_buffer_alloc(be, nw * sizeof(float));
    backend_buffer_t* b_wa = backend_buffer_alloc(be, nw * sizeof(float));
    backend_buffer_t* b_par = backend_buffer_alloc(be, 16 * sizeof(int));
    upload(b_g1, gout1, BT * OC * sizeof(float));
    upload(b_g2, gout2, BT * OC * sizeof(float));
    upload(b_i1, inp1, BT * C * sizeof(float));
    upload(b_i2, inp2, BT * C * sizeof(float));
    int params[3] = {BT, C, OC};
    upload(b_par, params, sizeof(params));

    backend_kernel_t* k = backend_kernel_create(be, name);
    if (!k) { fprintf(stderr, "%s not found\n", name); return 0; }

    int gx = tiled ? ((C + 31) / 32) * 256 : C;
    int gy = tiled ? (OC + 31) / 32 : OC;
    int tx = tiled ? 256 : C;
    int ty = tiled ? 1 : OC;

    float* w1 = malloc(nw * sizeof(float));
    float* w2 = malloc(nw * sizeof(float));
    float* wa = malloc(nw * sizeof(float));

    zero(b_w, nw * sizeof(float));
    backend_encoder_t* enc = backend_encode_begin(be);
    backend_buffer_t* bufs1[] = {b_g1, b_i1, b_w, b_par};
    backend_encode_dispatch(enc, k, bufs1, NULL, 4, gx, gy, 1, tx, ty, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);
    download(b_w, w1, nw * sizeof(float));

    zero(b_wa, nw * sizeof(float));
    enc = backend_encode_begin(be);
    backend_buffer_t* bufs2[] = {b_g2, b_i2, b_wa, b_par};
    backend_encode_dispatch(enc, k, bufs2, NULL, 4, gx, gy, 1, tx, ty, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);
    download(b_wa, w2, nw * sizeof(float));

    zero(b_w, nw * sizeof(float));
    enc = backend_encode_begin(be);
    backend_buffer_t* bufs3[] = {b_g1, b_i1, b_w, b_par};
    backend_encode_dispatch(enc, k, bufs3, NULL, 4, gx, gy, 1, tx, ty, 1);
    backend_buffer_t* bufs4[] = {b_g2, b_i2, b_w, b_par};
    backend_encode_dispatch(enc, k, bufs4, NULL, 4, gx, gy, 1, tx, ty, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);
    download(b_w, wa, nw * sizeof(float));

    float maxdiff = 0.0f;
    for (size_t i = 0; i < nw; i++) {
        float d = fabsf(wa[i] - (w1[i] + w2[i]));
        if (d > maxdiff) maxdiff = d;
    }
    int ok = maxdiff < 1e-5f;
    printf("%s: maxdiff=%.3g %s\n", name, maxdiff, ok ? "PASS" : "FAIL");

    free(gout1); free(gout2); free(inp1); free(inp2);
    free(w1); free(w2); free(wa);
    backend_buffer_free(b_g1); backend_buffer_free(b_g2);
    backend_buffer_free(b_i1); backend_buffer_free(b_i2);
    backend_buffer_free(b_w); backend_buffer_free(b_wa);
    backend_buffer_free(b_par);
    backend_kernel_destroy(k);
    return ok;
}

typedef struct {
    bm_model_t*  model;
    bm_trainer_t* trainer;
} train_case_t;

static int make_case(bm_context_t* ctx, const char* model_path, int S, int G,
                     train_case_t* c) {
    memset(c, 0, sizeof(*c));
    c->model = calloc(1, sizeof(bm_model_t));
    if (!c->model) return 0;
    bm_load_weights(c->model, model_path);
    c->model->precision = bm_get_supported_precision(ctx);
    bm_train_config_t cfg = {0};
    cfg.learning_rate = 1e-3f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.grad_clip = 0.0f;
    cfg.grad_accum_steps = G;
    cfg.warmup_steps = 0;
    cfg.max_steps = 1;
    cfg.use_master_weights = 1;
    cfg.seq_len = S;
    c->trainer = bmt_trainer_create(ctx, c->model, &cfg, S);
    return c->trainer != NULL;
}

static void free_case(train_case_t* c) {
    if (c->trainer) bm_destroy_trainer(c->trainer);
    if (c->model) bm_destroy_model(c->model);
    memset(c, 0, sizeof(*c));
}

typedef struct {
    int n;
    size_t total;
    size_t* ns;
    float* data;
} snapshot_t;

static float* state_ptr(bm_trainer_t* t, bmt_adamw_state_t* st, int* mapped_master) {
    *mapped_master = st->master_w != NULL;
    if (st->master_w) return (float*)backend_buffer_map(st->master_w);
    return (float*)backend_buffer_map(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
}

static void state_unmap(bm_trainer_t* t, bmt_adamw_state_t* st, int mapped_master) {
    if (mapped_master) backend_buffer_unmap(st->master_w);
    else backend_buffer_unmap(bmt_scheduler_get_buffer(t->sched, st->grad_tensor_id));
}

static int take_snapshot(bm_trainer_t* t, snapshot_t* s) {
    memset(s, 0, sizeof(*s));
    s->n = t->n_opt_states;
    s->ns = malloc((size_t)s->n * sizeof(size_t));
    if (!s->ns) return 0;
    s->total = 0;
    for (int i = 0; i < s->n; i++) {
        s->ns[i] = t->opt_states[i].n_params;
        s->total += s->ns[i];
    }
    s->data = malloc(s->total * sizeof(float));
    if (!s->data) return 0;
    size_t off = 0;
    for (int i = 0; i < s->n; i++) {
        int mm = 0;
        float* p = state_ptr(t, &t->opt_states[i], &mm);
        memcpy(s->data + off, p, s->ns[i] * sizeof(float));
        state_unmap(t, &t->opt_states[i], mm);
        off += s->ns[i];
    }
    return 1;
}

static void free_snapshot(snapshot_t* s) {
    free(s->ns);
    free(s->data);
    memset(s, 0, sizeof(*s));
}

static int compare_snapshot(bm_trainer_t* t, const snapshot_t* ref,
                            float* out_maxabs, double* out_cos) {
    if (t->n_opt_states != ref->n) return 0;
    size_t off = 0;
    float maxabs = 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < t->n_opt_states; i++) {
        if (t->opt_states[i].n_params != ref->ns[i]) return 0;
        int mm = 0;
        float* p = state_ptr(t, &t->opt_states[i], &mm);
        const float* q = ref->data + off;
        for (size_t j = 0; j < ref->ns[i]; j++) {
            float a = p[j], b = q[j];
            float d = fabsf(a - b);
            if (d > maxabs) maxabs = d;
            dot += (double)a * b;
            na += (double)a * a;
            nb += (double)b * b;
        }
        state_unmap(t, &t->opt_states[i], mm);
        off += ref->ns[i];
    }
    *out_maxabs = maxabs;
    *out_cos = dot / (sqrt(na) * sqrt(nb) + 1e-12);
    return 1;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "data/smollm2-135m";
    const int S = 8;
    const int seq[17] = {5, 11, 23, 41, 17, 3, 29, 37, 7, 13, 31, 43, 19, 2, 47, 53, 59};

    int ok = 1;
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }
    ok &= check_dw_kernel(be, "matmul_backward_w", 0);
    ok &= check_dw_kernel(be, "matmul_backward_w_tiled", 1);
    backend_destroy(be);

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "no context\n"); return 1; }

    snapshot_t ref;
    {
        train_case_t a;
        if (!make_case(ctx, model_path, S, 1, &a)) {
            fprintf(stderr, "cannot create B=2 case\n");
            bm_destroy(ctx);
            return 1;
        }
        float loss = bm_train_step(a.trainer, seq, seq + 1, 2, S);
        if (!isfinite(loss)) { fprintf(stderr, "non-finite loss\n"); ok = 0; }
        if (!take_snapshot(a.trainer, &ref)) { fprintf(stderr, "snapshot failed\n"); ok = 0; }
        free_case(&a);
    }
    {
        train_case_t b;
        if (!make_case(ctx, model_path, S, 2, &b)) {
            fprintf(stderr, "cannot create G=2 case\n");
            free_snapshot(&ref);
            bm_destroy(ctx);
            return 1;
        }
        float l0 = bm_train_step(b.trainer, seq, seq + 1, 1, S);
        float l1 = bm_train_step(b.trainer, seq + 8, seq + 9, 1, S);
        if (!isfinite(l0) || !isfinite(l1)) { fprintf(stderr, "non-finite loss\n"); ok = 0; }
        float maxabs = 0.0f;
        double cos = 0.0;
        if (ok && compare_snapshot(b.trainer, &ref, &maxabs, &cos)) {
            printf("B=2 vs G=2: maxabs=%.3g cos=%.9f\n", maxabs, cos);
            if (maxabs > 5e-3f || cos < 0.99999) ok = 0;
        } else if (ok) {
            fprintf(stderr, "weight snapshot mismatch\n");
            ok = 0;
        }
        free_case(&b);
    }

    free_snapshot(&ref);
    bm_destroy(ctx);
    printf("\n=== grad_accum: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
