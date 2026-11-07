/* masked_sft_test.c -- masked cross-entropy and padding-aware attention. */
#include "backend/backend.h"
#include "baremetal/graph.h"
#include "baremetal/scheduler.h"
#include "kernels/registry.h"
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/trainer.h"
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

static int test_xent(backend_ctx_t* be) {
    const int N = 4, V = 8;
    const int targets[4] = {1, 3, 5, 0};
    const uint8_t mask[4] = {1, 0, 1, 0};
    float logits[32];
    for (int i = 0; i < N * V; i++) logits[i] = sinf(0.13f * (float)(i + 1));

    int count = 0;
    for (int i = 0; i < N; i++) count += mask[i] ? 1 : 0;

    float exp_loss = 0.0f;
    float exp_grad[32];
    memset(exp_grad, 0, sizeof(exp_grad));
    for (int n = 0; n < N; n++) {
        if (!mask[n]) continue;
        const float* lr = logits + n * V;
        float mx = lr[0];
        for (int j = 1; j < V; j++) if (lr[j] > mx) mx = lr[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) sum += expf(lr[j] - mx);
        exp_loss += -(lr[targets[n]] - mx - logf(sum));
        for (int j = 0; j < V; j++) {
            float p = expf(lr[j] - mx) / sum;
            exp_grad[n * V + j] = (p - (j == targets[n] ? 1.0f : 0.0f)) / (float)count;
        }
    }
    exp_loss /= (float)count;

    backend_buffer_t* b_logits = backend_buffer_alloc(be, N * V * sizeof(float));
    backend_buffer_t* b_tgt = backend_buffer_alloc(be, N * sizeof(int));
    backend_buffer_t* b_grad = backend_buffer_alloc(be, N * V * sizeof(float));
    backend_buffer_t* b_mask = backend_buffer_alloc(be, N);
    backend_buffer_t* b_par = backend_buffer_alloc(be, 16 * sizeof(int));
    upload(b_logits, logits, sizeof(logits));
    upload(b_tgt, targets, sizeof(targets));
    upload(b_mask, mask, sizeof(mask));
    zero(b_grad, N * V * sizeof(float));
    int par[3] = {N, V, count};
    upload(b_par, par, sizeof(par));

    backend_kernel_t* k = backend_kernel_create(be, "xent_backward_masked");
    if (!k) { fprintf(stderr, "xent_backward_masked missing\n"); return 0; }
    backend_encoder_t* enc = backend_encode_begin(be);
    backend_buffer_t* bufs[] = {b_logits, b_tgt, b_grad, b_mask, b_par};
    backend_encode_dispatch(enc, k, bufs, NULL, 5, N, 1, 1, N, 1, 1);
    backend_encode_commit(enc);
    backend_encode_wait(enc);

    float got[32];
    download(b_grad, got, sizeof(got));
    float maxdiff = 0.0f;
    for (int i = 0; i < N * V; i++) {
        float d = fabsf(got[i] - exp_grad[i]);
        if (d > maxdiff) maxdiff = d;
    }
    float loss = 0.0f;
    for (int n = 0; n < N; n++) {
        if (!mask[n]) continue;
        const float* lr = logits + n * V;
        float mx = lr[0];
        for (int j = 1; j < V; j++) if (lr[j] > mx) mx = lr[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) sum += expf(lr[j] - mx);
        loss += -(lr[targets[n]] - mx - logf(sum));
    }
    loss /= (float)count;
    printf("xent masked loss: %.6f (want %.6f), maxdiff %.3g\n", loss, exp_loss, maxdiff);
    int ok = fabsf(loss - exp_loss) < 1e-6f && maxdiff < 1e-6f;
    for (int n = 0; n < N; n++) if (!mask[n])
        for (int j = 0; j < V; j++) if (got[n * V + j] != 0.0f) ok = 0;
    check(ok, "masked CE loss and gradients");

    backend_buffer_free(b_logits); backend_buffer_free(b_tgt); backend_buffer_free(b_grad);
    backend_buffer_free(b_mask); backend_buffer_free(b_par);
    backend_kernel_destroy(k);
    return ok;
}

static int test_xent_scheduler(backend_ctx_t* be, const float* logits,
                               const int* targets, const uint8_t* mask,
                               int N, int V, float exp_loss,
                               const float* exp_grad) {
    bmt_graph_t* g = bmt_graph_create();
    int id = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){N, V});
    bmk_registry_t* reg = bmk_registry_create(be);
    bmt_scheduler_t* sched = bmt_scheduler_create(be, reg, g, N, 1, 1);
    if (!sched) { fprintf(stderr, "scheduler failed\n"); return 0; }
    bmt_scheduler_set_input(sched, id, logits, (size_t)N * V * sizeof(float));
    float loss = bmt_scheduler_xent_backward_masked(sched, id, targets, mask, N, V);
    backend_buffer_t* gb = bmt_scheduler_get_grad_buffer(sched, id);
    float got[32];
    download(gb, got, (size_t)N * V * sizeof(float));
    float maxdiff = 0.0f;
    for (int i = 0; i < N * V; i++) {
        float d = fabsf(got[i] - exp_grad[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("scheduler masked loss: %.6f (want %.6f), maxdiff %.3g\n", loss, exp_loss, maxdiff);
    int ok = fabsf(loss - exp_loss) < 1e-6f && maxdiff < 1e-6f;
    check(ok, "scheduler masked CE");
    bmt_scheduler_destroy(sched);
    bmk_registry_destroy(reg);
    bmt_graph_destroy(g);
    return ok;
}

static void fill_attn(float* q, float* k, float* v, int S, int HD) {
    for (int i = 0; i < S; i++) {
        for (int d = 0; d < HD; d++) {
            q[i * HD + d] = sinf(0.11f * (float)(i * HD + d + 1));
            k[i * HD + d] = cosf(0.07f * (float)(i * HD + d + 1));
            v[i * HD + d] = sinf(0.05f * (float)(i * HD + d + 2));
        }
    }
}

static int test_attention(backend_ctx_t* be) {
    const int NH = 1, NKV = 1, HD = 4, S = 4, SV = 3;
    const uint8_t mask[4] = {1, 1, 1, 0};
    float q[16], k[16], v[16];
    fill_attn(q, k, v, S, HD);
    float scale = 1.0f / sqrtf((float)HD);

    backend_buffer_t* b_q = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_k = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_v = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_o = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_om = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_mask = backend_buffer_alloc(be, S);
    backend_buffer_t* b_par = backend_buffer_alloc(be, 16 * sizeof(int));
    backend_buffer_t* b_scale = backend_buffer_alloc(be, sizeof(float));
    upload(b_scale, &scale, sizeof(float));
    upload(b_q, q, sizeof(q)); upload(b_k, k, sizeof(k)); upload(b_v, v, sizeof(v));
    upload(b_mask, mask, sizeof(mask));
    int par[5] = {NH, SV, HD, NKV, 1};
    upload(b_par, par, sizeof(par));

    backend_kernel_t* kf = backend_kernel_create(be, "attention_forward_seq");
    backend_kernel_t* kfm = backend_kernel_create(be, "attention_forward_seq_masked");
    if (!kf || !kfm) { fprintf(stderr, "attention forward kernels missing\n"); return 0; }

    backend_encoder_t* enc = backend_encode_begin(be);
    backend_buffer_t* b1[] = {b_q, b_k, b_v, b_o, b_par, b_scale};
    backend_encode_dispatch(enc, kf, b1, NULL, 6, SV * NH, 1, 1, HD, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    int par4[5] = {NH, S, HD, NKV, 1};
    upload(b_par, par4, sizeof(par4));
    enc = backend_encode_begin(be);
    backend_buffer_t* b2[] = {b_q, b_k, b_v, b_om, b_par, b_scale, b_mask};
    backend_encode_dispatch(enc, kfm, b2, NULL, 7, S * NH, 1, 1, HD, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    float o[16], om[16];
    download(b_o, o, sizeof(o));
    download(b_om, om, sizeof(om));
    float maxdiff = 0.0f;
    for (int i = 0; i < SV; i++)
        for (int d = 0; d < HD; d++) {
            float diff = fabsf(o[i * HD + d] - om[i * HD + d]);
            if (diff > maxdiff) maxdiff = diff;
        }
    for (int d = 0; d < HD; d++) if (om[SV * HD + d] != 0.0f) maxdiff = 1e9f;
    printf("attention masked forward maxdiff: %.3g\n", maxdiff);
    check(maxdiff < 1e-6f, "masked forward matches unpadded");

    /* backward */
    float gout[16];
    for (int i = 0; i < S; i++)
        for (int d = 0; d < HD; d++)
            gout[i * HD + d] = (i < SV) ? cosf(0.09f * (float)(i * HD + d + 1)) : 0.0f;

    backend_buffer_t* b_g = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gq = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gk = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gv = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gqm = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gkm = backend_buffer_alloc(be, S * HD * sizeof(float));
    backend_buffer_t* b_gvm = backend_buffer_alloc(be, S * HD * sizeof(float));
    upload(b_g, gout, sizeof(gout));
    zero(b_gq, S * HD * sizeof(float)); zero(b_gk, S * HD * sizeof(float)); zero(b_gv, S * HD * sizeof(float));
    zero(b_gqm, S * HD * sizeof(float)); zero(b_gkm, S * HD * sizeof(float)); zero(b_gvm, S * HD * sizeof(float));

    backend_kernel_t* kb = backend_kernel_create(be, "attention_backward");
    backend_kernel_t* kbm = backend_kernel_create(be, "attention_backward_masked");
    if (!kb || !kbm) { fprintf(stderr, "attention backward kernels missing\n"); return 0; }

    upload(b_par, par, sizeof(par));
    enc = backend_encode_begin(be);
    backend_buffer_t* b3[] = {b_g, b_q, b_k, b_v, b_gq, b_gk, b_gv, b_par, b_scale};
    backend_encode_dispatch(enc, kb, b3, NULL, 9, SV * NH, 1, 1, 32, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    upload(b_par, par4, sizeof(par4));
    enc = backend_encode_begin(be);
    backend_buffer_t* b4[] = {b_g, b_q, b_k, b_v, b_gqm, b_gkm, b_gvm, b_par, b_scale, b_mask};
    backend_encode_dispatch(enc, kbm, b4, NULL, 10, S * NH, 1, 1, 32, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    float gq[16], gk[16], gv[16], gqm[16], gkm[16], gvm[16];
    download(b_gq, gq, sizeof(gq)); download(b_gk, gk, sizeof(gk)); download(b_gv, gv, sizeof(gv));
    download(b_gqm, gqm, sizeof(gqm)); download(b_gkm, gkm, sizeof(gkm)); download(b_gvm, gvm, sizeof(gvm));

    float bmax = 0.0f;
    for (int i = 0; i < SV; i++)
        for (int d = 0; d < HD; d++) {
            float dq = fabsf(gq[i * HD + d] - gqm[i * HD + d]);
            if (dq > bmax) bmax = dq;
        }
    for (int i = 0; i < SV; i++)
        for (int d = 0; d < HD; d++) {
            float dk = fabsf(gk[i * HD + d] - gkm[i * HD + d]);
            float dv = fabsf(gv[i * HD + d] - gvm[i * HD + d]);
            if (dk > bmax) bmax = dk;
            if (dv > bmax) bmax = dv;
        }
    for (int d = 0; d < HD; d++) if (gqm[SV * HD + d] != 0.0f) bmax = 1e9f;
    printf("attention masked backward maxdiff: %.3g\n", bmax);
    check(bmax < 1e-4f, "masked backward matches unpadded");

    int ok = maxdiff < 1e-6f && bmax < 1e-4f;
    backend_buffer_free(b_q); backend_buffer_free(b_k); backend_buffer_free(b_v);
    backend_buffer_free(b_o); backend_buffer_free(b_om); backend_buffer_free(b_mask);
    backend_buffer_free(b_par); backend_buffer_free(b_scale); backend_buffer_free(b_g);
    backend_buffer_free(b_gq); backend_buffer_free(b_gk); backend_buffer_free(b_gv);
    backend_buffer_free(b_gqm); backend_buffer_free(b_gkm); backend_buffer_free(b_gvm);
    backend_kernel_destroy(kf); backend_kernel_destroy(kfm);
    backend_kernel_destroy(kb); backend_kernel_destroy(kbm);
    return ok;
}

static int test_trainer_masks(void) {
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) return 0;
    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, "data/smollm2-135m");
    m->precision = bm_get_supported_precision(ctx);
    bm_train_config_t cfg = {0};
    cfg.learning_rate = 0.0f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.grad_accum_steps = 1;
    cfg.use_master_weights = 1;
    bm_trainer_t* t = bmt_trainer_create(ctx, m, &cfg, 8);
    if (!t) { bm_destroy_model(m); bm_destroy(ctx); return 0; }

    int inputs[16], targets[16];
    for (int i = 0; i < 16; i++) { inputs[i] = 5 + i; targets[i] = 6 + i; }
    uint8_t ones[16];
    memset(ones, 1, sizeof(ones));
    float l_all = bm_train_step_masked(t, inputs, targets, NULL, NULL, 1, 8);
    float l_attn = bm_train_step_masked(t, inputs, targets, ones, NULL, 1, 8);
    float l_loss = bm_train_step_masked(t, inputs, targets, NULL, ones, 1, 8);
    float l_masked_ones = bm_train_step_masked(t, inputs, targets, ones, ones, 1, 8);
    printf("trainer loss: unmasked %.6f attn-only %.6f loss-only %.6f both %.6f\n",
           l_all, l_attn, l_loss, l_masked_ones);
    check(fabsf(l_all - l_attn) < 1e-5f, "attention all-ones matches unmasked");
    check(fabsf(l_all - l_loss) < 1e-5f, "loss all-ones matches unmasked");
    check(fabsf(l_all - l_masked_ones) < 1e-5f, "all-ones masks match unmasked");

    uint8_t loss_mask[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    float l_part = bm_train_step_masked(t, inputs, targets, ones, loss_mask, 1, 8);
    check(isfinite(l_part), "partial loss mask is finite");

    uint8_t attn2[16];
    uint8_t loss2[16];
    for (int i = 0; i < 16; i++) {
        attn2[i] = (i % 8) < 6 ? 1 : 0;
        loss2[i] = (i % 8) >= 4 && (i % 8) < 6 ? 1 : 0;
    }
    float l_b2 = bm_train_step_masked(t, inputs, targets, attn2, loss2, 2, 8);
    check(isfinite(l_b2), "B=2 masked step is finite");

    bm_destroy_trainer(t);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return 1;
}

int main(void) {
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }
    test_xent(be);
    {
        const int N = 4, V = 8;
        const int targets[4] = {1, 3, 5, 0};
        const uint8_t mask[4] = {1, 0, 1, 0};
        float logits[32];
        for (int i = 0; i < N * V; i++) logits[i] = sinf(0.13f * (float)(i + 1));
        int count = 0;
        for (int i = 0; i < N; i++) count += mask[i] ? 1 : 0;
        float exp_loss = 0.0f, exp_grad[32];
        memset(exp_grad, 0, sizeof(exp_grad));
        for (int n = 0; n < N; n++) {
            if (!mask[n]) continue;
            const float* lr = logits + n * V;
            float mx = lr[0];
            for (int j = 1; j < V; j++) if (lr[j] > mx) mx = lr[j];
            float sum = 0.0f;
            for (int j = 0; j < V; j++) sum += expf(lr[j] - mx);
            exp_loss += -(lr[targets[n]] - mx - logf(sum));
            for (int j = 0; j < V; j++) {
                float p = expf(lr[j] - mx) / sum;
                exp_grad[n * V + j] = (p - (j == targets[n] ? 1.0f : 0.0f)) / (float)count;
            }
        }
        exp_loss /= (float)count;
        test_xent_scheduler(be, logits, targets, mask, N, V, exp_loss, exp_grad);
    }
    test_attention(be);
    backend_destroy(be);
    test_trainer_masks();
    printf("\n=== masked_sft: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
