/* train_attn_tiled_test.c -- tiled training attention matches the old kernels. */
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void upload(backend_buffer_t* b, const void* p, size_t n) {
    memcpy(backend_buffer_map(b), p, n); backend_buffer_unmap(b);
}
static void download(backend_buffer_t* b, void* p, size_t n) {
    memcpy(p, backend_buffer_map(b), n); backend_buffer_unmap(b);
}
static void zero(backend_buffer_t* b) {
    memset(backend_buffer_map(b), 0, backend_buffer_size(b)); backend_buffer_unmap(b);
}
static float maxdiff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int NH = 4, NKV = 2, HD = 64, kv_mul = 2, MAXS = 256;
    backend_ctx_t* be = backend_create();
    if (!be) return 1;
    backend_kernel_t* ft = backend_kernel_create(be, "attention_forward_train_tiled");
    backend_kernel_t* fo = backend_kernel_create(be, "attention_forward_seq");
    backend_kernel_t* fm = backend_kernel_create(be, "attention_forward_seq_masked");
    backend_kernel_t* bt = backend_kernel_create(be, "attention_backward_tiled");
    backend_kernel_t* bo = backend_kernel_create(be, "attention_backward");
    backend_kernel_t* kbm = backend_kernel_create(be, "attention_backward_masked");
    if (!ft || !fo || !fm || !bt || !bo || !kbm) {
        fprintf(stderr, "missing training attention kernels\n");
        return 1;
    }

    float* Q = malloc((size_t)MAXS * NH * HD * sizeof(float));
    float* K = malloc((size_t)MAXS * NKV * HD * sizeof(float));
    float* V = malloc((size_t)MAXS * NKV * HD * sizeof(float));
    float* G = malloc((size_t)MAXS * NH * HD * sizeof(float));
    unsigned char* mask = malloc(MAXS);
    if (!Q || !K || !V || !G || !mask) return 1;
    for (int i = 0; i < MAXS * NH * HD; i++) Q[i] = sinf(0.011f * (float)(i + 1));
    for (int i = 0; i < MAXS * NH * HD; i++) G[i] = cosf(0.007f * (float)(i + 2));
    for (int i = 0; i < MAXS * NKV * HD; i++) K[i] = cosf(0.013f * (float)(i + 1));
    for (int i = 0; i < MAXS * NKV * HD; i++) V[i] = sinf(0.017f * (float)(i + 3));

    backend_buffer_t* bq = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bk = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bv = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bg = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bo1 = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bo2 = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bq1 = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bq2 = backend_buffer_alloc(be, (size_t)MAXS * NH * HD * sizeof(float));
    backend_buffer_t* bk1 = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bk2 = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bv1 = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bv2 = backend_buffer_alloc(be, (size_t)MAXS * NKV * HD * sizeof(float));
    backend_buffer_t* bp = backend_buffer_alloc(be, 8 * sizeof(int));
    backend_buffer_t* bs = backend_buffer_alloc(be, sizeof(float));
    backend_buffer_t* bmask = backend_buffer_alloc(be, MAXS);
    upload(bq, Q, (size_t)MAXS * NH * HD * sizeof(float));
    upload(bk, K, (size_t)MAXS * NKV * HD * sizeof(float));
    upload(bv, V, (size_t)MAXS * NKV * HD * sizeof(float));
    upload(bg, G, (size_t)MAXS * NH * HD * sizeof(float));
    float scale = 1.0f / sqrtf((float)HD);
    upload(bs, &scale, sizeof(scale));

    int sizes[] = {64, 128, 256};
    float max_err = 0.0f;
    int fails = 0;
    for (int si = 0; si < 3; si++) {
        int S = sizes[si];
        for (int masked = 0; masked <= 1; masked++) {
            for (int i = 0; i < S; i++) mask[i] = masked ? (i < S - 13) : 1;
            upload(bmask, mask, S);
            int p[8] = {NH, S, HD, NKV, kv_mul, masked, 0, 0};
            upload(bp, p, sizeof(p));

            int gt = ((S + 31) / 32) * NH * 256;
            int go = S * NH;

            /* forward */
            zero(bo1); zero(bo2);
            backend_encoder_t* enc = backend_encode_begin(be);
            backend_buffer_t* f1[] = {bq, bk, bv, bo1, bp, bs, bmask};
            backend_encode_dispatch(enc, ft, f1, NULL, 7, gt, 1, 1, 256, 1, 1);
            backend_encode_commit(enc); backend_encode_wait(enc);
            enc = backend_encode_begin(be);
            backend_buffer_t* f2[] = {bq, bk, bv, bo2, bp, bs, bmask};
            backend_encode_dispatch(enc, masked ? fm : fo, f2, NULL, masked ? 7 : 6,
                                    go, 1, 1, 32, 1, 1);
            backend_encode_commit(enc); backend_encode_wait(enc);
            float* a = malloc((size_t)S * NH * HD * sizeof(float));
            float* b = malloc((size_t)S * NH * HD * sizeof(float));
            download(bo1, a, (size_t)S * NH * HD * sizeof(float));
            download(bo2, b, (size_t)S * NH * HD * sizeof(float));
            float e = maxdiff(a, b, S * NH * HD);
            printf("S=%d mask=%d fwd maxabs=%.3g %s\n", S, masked, e, e < 2e-5f ? "OK" : "FAIL");
            if (e >= 2e-5f) fails++;
            free(a); free(b);

            /* backward */
            zero(bq1); zero(bq2); zero(bk1); zero(bk2); zero(bv1); zero(bv2);
            enc = backend_encode_begin(be);
            backend_buffer_t* t1[] = {bg, bq, bk, bv, bq1, bk1, bv1, bp, bs, bmask};
            backend_encode_dispatch(enc, bt, t1, NULL, 10, gt, 1, 1, 256, 1, 1);
            backend_encode_commit(enc); backend_encode_wait(enc);
            enc = backend_encode_begin(be);
            backend_buffer_t* t2[] = {bg, bq, bk, bv, bq2, bk2, bv2, bp, bs, bmask};
            backend_encode_dispatch(enc, masked ? kbm : bo, t2, NULL, masked ? 10 : 9,
                                    go, 1, 1, 32, 1, 1);
            backend_encode_commit(enc); backend_encode_wait(enc);
            float* ga = malloc((size_t)S * NH * HD * sizeof(float));
            float* gb = malloc((size_t)S * NH * HD * sizeof(float));
            download(bq1, ga, (size_t)S * NH * HD * sizeof(float));
            download(bq2, gb, (size_t)S * NH * HD * sizeof(float));
            e = maxdiff(ga, gb, S * NH * HD);
            printf("S=%d mask=%d gQ maxabs=%.3g %s\n", S, masked, e, e < 2e-4f ? "OK" : "FAIL");
            if (e >= 2e-4f) fails++;
            free(ga); free(gb);

            ga = malloc((size_t)S * NKV * HD * sizeof(float));
            gb = malloc((size_t)S * NKV * HD * sizeof(float));
            download(bk1, ga, (size_t)S * NKV * HD * sizeof(float));
            download(bk2, gb, (size_t)S * NKV * HD * sizeof(float));
            e = maxdiff(ga, gb, S * NKV * HD);
            if (e > max_err) max_err = e;
            printf("S=%d mask=%d gK maxabs=%.3g %s\n", S, masked, e, e < 2e-4f ? "OK" : "FAIL");
            if (e >= 2e-4f) fails++;
            download(bv1, ga, (size_t)S * NKV * HD * sizeof(float));
            download(bv2, gb, (size_t)S * NKV * HD * sizeof(float));
            e = maxdiff(ga, gb, S * NKV * HD);
            if (e > max_err) max_err = e;
            printf("S=%d mask=%d gV maxabs=%.3g %s\n", S, masked, e, e < 2e-4f ? "OK" : "FAIL");
            if (e >= 2e-4f) fails++;
            free(ga); free(gb);
        }
    }
    printf("\n=== train_attn_tiled: %s (max %.3g) ===\n", fails ? "FAIL" : "PASS", max_err);
    return fails ? 1 : 0;
}
