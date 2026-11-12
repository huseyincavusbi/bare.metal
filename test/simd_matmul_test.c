/* simd_matmul_test.c -- simdgroup bf16 matmul matches the scalar tile on edge shapes. */
#include "backend/backend.h"
#include "baremetal/types.h"
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

static double cosine(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int main(void) {
    const int BT = 37, C = 53, OC = 70;
    const size_t in_n = (size_t)BT * C;
    const size_t w_n = (size_t)OC * C;
    const size_t out_n = (size_t)BT * OC;

    backend_ctx_t* be = backend_create();
    if (!be) return 1;
    backend_kernel_t* k_simd = backend_kernel_create(be, "matmul_forward_bf16_simd");
    backend_kernel_t* k_tiled = backend_kernel_create(be, "matmul_forward_bf16_tiled64");
    if (!k_simd || !k_tiled) {
        fprintf(stderr, "missing matmul kernels\n");
        return 1;
    }

    float* in = malloc(in_n * sizeof(float));
    float* wf = malloc(w_n * sizeof(float));
    bm_bf16_t* w = malloc(w_n * sizeof(bm_bf16_t));
    float* ref = malloc(out_n * sizeof(float));
    float* out_simd = malloc(out_n * sizeof(float));
    float* out_tiled = malloc(out_n * sizeof(float));
    if (!in || !wf || !w || !ref || !out_simd || !out_tiled) return 1;

    for (size_t i = 0; i < in_n; i++)
        in[i] = (float)(((int)(i * 7) % 17) - 8) * 0.25f;
    for (size_t i = 0; i < w_n; i++)
        wf[i] = (float)(((int)(i * 11) % 13) - 6) * 0.5f;
    bm_f32_to_bf16_array(wf, w, w_n);

    for (int m = 0; m < BT; m++) {
        for (int n = 0; n < OC; n++) {
            float sum = 0.0f;
            for (int k = 0; k < C; k++)
                sum += in[(size_t)m * C + k] * bm_bf16_to_f32(w[(size_t)n * C + k]);
            ref[(size_t)m * OC + n] = sum;
        }
    }

    backend_buffer_t* b_in = backend_buffer_alloc(be, in_n * sizeof(float));
    backend_buffer_t* b_w = backend_buffer_alloc(be, w_n * sizeof(bm_bf16_t));
    backend_buffer_t* b_bias = backend_buffer_alloc(be, 16 * sizeof(float));
    backend_buffer_t* b_simd = backend_buffer_alloc(be, out_n * sizeof(float));
    backend_buffer_t* b_tiled = backend_buffer_alloc(be, out_n * sizeof(float));
    backend_buffer_t* b_p = backend_buffer_alloc(be, 4 * sizeof(int));
    upload(b_in, in, in_n * sizeof(float));
    upload(b_w, w, w_n * sizeof(bm_bf16_t));
    float zero_bias[16] = {0};
    upload(b_bias, zero_bias, sizeof(zero_bias));
    int p[4] = {BT, C, OC, 0};
    upload(b_p, p, sizeof(p));

    int gtx = ((OC + 63) / 64) * 128;
    int gty = (BT + 63) / 64;

    backend_encoder_t* enc = backend_encode_begin(be);
    backend_buffer_t* bufs[] = {b_in, b_w, b_bias, b_simd, b_p};
    backend_encode_dispatch(enc, k_simd, bufs, NULL, 5, gtx, gty, 1, 128, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    enc = backend_encode_begin(be);
    backend_buffer_t* bufs2[] = {b_in, b_w, b_bias, b_tiled, b_p};
    backend_encode_dispatch(enc, k_tiled, bufs2, NULL, 5, gtx, gty, 1, 128, 1, 1);
    backend_encode_commit(enc); backend_encode_wait(enc);

    download(b_simd, out_simd, out_n * sizeof(float));
    download(b_tiled, out_tiled, out_n * sizeof(float));

    float max_simd = 0.0f, max_tiled = 0.0f;
    for (size_t i = 0; i < out_n; i++) {
        float d = fabsf(out_simd[i] - ref[i]);
        if (d > max_simd) max_simd = d;
        d = fabsf(out_tiled[i] - ref[i]);
        if (d > max_tiled) max_tiled = d;
    }
    double c_simd = cosine(out_simd, ref, (int)out_n);
    double c_tiled = cosine(out_tiled, ref, (int)out_n);
    int ok = max_simd < 1e-2f && max_tiled < 1e-2f
             && c_simd > 0.99999 && c_tiled > 0.99999;
    printf("simd  maxabs=%.3g cos=%.7f\n", max_simd, c_simd);
    printf("tiled maxabs=%.3g cos=%.7f\n", max_tiled, c_tiled);
    printf("\n=== simd_matmul: %s ===\n", ok ? "PASS" : "FAIL");

    backend_buffer_free(b_in); backend_buffer_free(b_w); backend_buffer_free(b_bias);
    backend_buffer_free(b_simd); backend_buffer_free(b_tiled); backend_buffer_free(b_p);
    backend_kernel_destroy(k_simd); backend_kernel_destroy(k_tiled);
    backend_destroy(be);
    free(in); free(wf); free(w); free(ref); free(out_simd); free(out_tiled);
    return ok ? 0 : 1;
}
