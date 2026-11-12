/* attn_prefill_test.c -- tiled prefill attention matches the warp kernel at block edges. */
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures;

static void upload(backend_buffer_t* b, const void* p, size_t n) {
    memcpy(backend_buffer_map(b), p, n); backend_buffer_unmap(b);
}

static void download(backend_buffer_t* b, void* p, size_t n) {
    memcpy(p, backend_buffer_map(b), n); backend_buffer_unmap(b);
}

int main(void) {
    const int NH = 4, NKV = 2, HD = 64, MS = 512;
    const int kv_mul = NH / NKV;
    backend_ctx_t* be = backend_create();
    if (!be) return 1;
    backend_kernel_t* kt = backend_kernel_create(be, "attention_forward_prefill_tiled");
    backend_kernel_t* kw = backend_kernel_create(be, "attention_forward_prefill_warp");
    if (!kt || !kw) { fprintf(stderr, "missing prefill attention kernels\n"); return 1; }

    float* q = malloc((size_t)MS * NH * HD * sizeof(float));
    float* k = malloc((size_t)MS * NKV * HD * sizeof(float));
    float* v = malloc((size_t)MS * NKV * HD * sizeof(float));
    float* a = malloc((size_t)MS * NH * HD * sizeof(float));
    float* b = malloc((size_t)MS * NH * HD * sizeof(float));
    if (!q || !k || !v || !a || !b) return 1;
    for (int i = 0; i < MS * NH * HD; i++) q[i] = sinf(0.013f * (float)(i + 1));
    for (int i = 0; i < MS * NKV * HD; i++) k[i] = cosf(0.017f * (float)(i + 1));
    for (int i = 0; i < MS * NKV * HD; i++) v[i] = sinf(0.019f * (float)(i + 3));

    backend_buffer_t* bq = backend_buffer_alloc(be, (size_t)MS * NH * HD * sizeof(float));
    backend_buffer_t* bk = backend_buffer_alloc(be, (size_t)MS * NKV * HD * sizeof(float));
    backend_buffer_t* bv = backend_buffer_alloc(be, (size_t)MS * NKV * HD * sizeof(float));
    backend_buffer_t* bo1 = backend_buffer_alloc(be, (size_t)MS * NH * HD * sizeof(float));
    backend_buffer_t* bo2 = backend_buffer_alloc(be, (size_t)MS * NH * HD * sizeof(float));
    backend_buffer_t* bp = backend_buffer_alloc(be, 8 * sizeof(int));
    backend_buffer_t* bs = backend_buffer_alloc(be, sizeof(float));
    upload(bq, q, (size_t)MS * NH * HD * sizeof(float));
    upload(bk, k, (size_t)MS * NKV * HD * sizeof(float));
    upload(bv, v, (size_t)MS * NKV * HD * sizeof(float));
    float scale = 1.0f / sqrtf((float)HD);
    upload(bs, &scale, sizeof(scale));

    int sizes[] = {1, 7, 17, 31, 32, 33, 64, 65, 127, 128, 129};
    for (int si = 0; si < 11; si++) {
        int S = sizes[si];
        int p[8] = {NH, S, HD, NKV, kv_mul, 0, 0, MS};
        upload(bp, p, sizeof(p));

        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* b1[] = {bq, bk, bv, bo1, bp, bs};
        backend_encode_dispatch(enc, kt, b1, NULL, 6,
                                ((S + 31) / 32) * NH * 256, 1, 1, 256, 1, 1);
        backend_encode_commit(enc); backend_encode_wait(enc);

        enc = backend_encode_begin(be);
        backend_buffer_t* b2[] = {bq, bk, bv, bo2, bp, bs};
        backend_encode_dispatch(enc, kw, b2, NULL, 6,
                                ((S + 7) / 8) * NH * 256, 1, 1, 256, 1, 1);
        backend_encode_commit(enc); backend_encode_wait(enc);

        download(bo1, a, (size_t)S * NH * HD * sizeof(float));
        download(bo2, b, (size_t)S * NH * HD * sizeof(float));
        float maxabs = 0.0f;
        for (int i = 0; i < S * NH * HD; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > maxabs) maxabs = d;
        }
        printf("S=%d maxabs=%.3g %s\n", S, maxabs, maxabs < 1e-5f ? "PASS" : "FAIL");
        if (maxabs >= 1e-5f) failures++;
    }

    backend_buffer_free(bq); backend_buffer_free(bk); backend_buffer_free(bv);
    backend_buffer_free(bo1); backend_buffer_free(bo2); backend_buffer_free(bp);
    backend_buffer_free(bs);
    backend_kernel_destroy(kt); backend_kernel_destroy(kw);
    backend_destroy(be);
    free(q); free(k); free(v); free(a); free(b);
    printf("\n=== attn_prefill: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
