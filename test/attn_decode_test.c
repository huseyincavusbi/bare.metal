/* attn_decode_test.c -- tiled decode attention matches the legacy flash kernel. */
#include "backend/backend.h"
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

static void upload(backend_buffer_t* b, const void* p, size_t n) {
    memcpy(backend_buffer_map(b), p, n); backend_buffer_unmap(b);
}

static void download(backend_buffer_t* b, void* p, size_t n) {
    memcpy(p, backend_buffer_map(b), n); backend_buffer_unmap(b);
}

int main(void) {
    const int NH = 2, HD = 64, NKV = 2, KV = 128, MS = 8192;
    backend_ctx_t* be = backend_create();
    if (!be) return 1;
    backend_kernel_t* k_tiled = backend_kernel_create(be, "attention_forward");
    backend_kernel_t* k_flash = backend_kernel_create(be, "attention_forward_flash");
    if (!k_tiled || !k_flash) { fprintf(stderr, "missing attention kernels\n"); return 1; }

    float q[NH * HD];
    size_t kv_n = (size_t)2 * MS * KV;
    float* kv = malloc(kv_n * sizeof(float));
    if (!kv) return 1;
    for (int i = 0; i < NH * HD; i++) q[i] = sinf(0.07f * (float)(i + 1));
    for (size_t i = 0; i < kv_n; i++) kv[i] = cosf(0.03f * (float)(i + 1));

    backend_buffer_t* bq = backend_buffer_alloc(be, sizeof(q));
    backend_buffer_t* bkv = backend_buffer_alloc(be, kv_n * sizeof(float));
    backend_buffer_t* bo = backend_buffer_alloc(be, sizeof(q));
    backend_buffer_t* bod = backend_buffer_alloc(be, sizeof(q));
    backend_buffer_t* bp = backend_buffer_alloc(be, 16 * sizeof(int));
    backend_buffer_t* bs = backend_buffer_alloc(be, sizeof(float));
    upload(bq, q, sizeof(q)); upload(bkv, kv, kv_n * sizeof(float));
    float scale = 1.0f / sqrtf((float)HD);
    upload(bs, &scale, sizeof(scale));

    int sizes[] = {1, 17, 256, 1024, 2048, 4096, 8192};
    for (int si = 0; si < 7; si++) {
        int S = sizes[si];
        int p[8] = {NH, HD, NKV, KV, S, 1, 0, MS};
        upload(bp, p, sizeof(p));
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* b1[] = {bq, bkv, bo, bp, bs};
        backend_encode_dispatch(enc, k_tiled, b1, NULL, 5, NH * HD, 1, 1, HD, 1, 1);
        backend_encode_commit(enc); backend_encode_wait(enc);

        enc = backend_encode_begin(be);
        backend_buffer_t* b2[] = {bq, bkv, bod, bp, bs};
        backend_encode_dispatch(enc, k_flash, b2, NULL, 5, NH * HD, 1, 1, HD, 1, 1);
        backend_encode_commit(enc); backend_encode_wait(enc);

        float a[NH * HD], b[NH * HD];
        download(bo, a, sizeof(a)); download(bod, b, sizeof(b));
        float maxabs = 0.0f;
        for (int i = 0; i < NH * HD; i++) {
            float d = fabsf(a[i] - b[i]);
            if (d > maxabs) maxabs = d;
        }
        printf("S=%d maxabs=%.3g\n", S, maxabs);
        check(maxabs < 1e-5f, "decode attention matches");
    }

    backend_buffer_free(bq); backend_buffer_free(bkv);
    backend_buffer_free(bo); backend_buffer_free(bod);
    backend_buffer_free(bp); backend_buffer_free(bs);
    backend_kernel_destroy(k_tiled); backend_kernel_destroy(k_flash);
    backend_destroy(be);
    free(kv);
    printf("\n=== attn_decode: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
