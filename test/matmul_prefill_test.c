#include "backend/backend.h"
#include "baremetal/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static unsigned rng_state = 12345;
static float rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static double cosine(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}

static int run_case(backend_ctx_t* be, backend_kernel_t* ref, backend_kernel_t* tiled,
                    int BT, int C, int OC, int fp16) {
    size_t in_elems = (size_t)BT * C;
    size_t w_elems = (size_t)OC * C;
    size_t out_elems = (size_t)BT * OC;

    float* inp = malloc(in_elems * sizeof(float));
    float* bias = calloc((size_t)OC, sizeof(float));
    float* ref_out = calloc(out_elems, sizeof(float));
    float* tiled_out = calloc(out_elems, sizeof(float));
    void* wsrc = malloc(w_elems * (fp16 ? sizeof(bm_fp16_t) : sizeof(bm_bf16_t)));
    if (!inp || !bias || !ref_out || !tiled_out || !wsrc) return 1;

    for (size_t i = 0; i < in_elems; i++) inp[i] = rnd();
    for (size_t i = 0; i < w_elems; i++) {
        float v = rnd() * 0.1f;
        if (fp16) {
            __fp16 h = (__fp16)v;
            memcpy(&((bm_fp16_t*)wsrc)[i], &h, sizeof(h));
        } else {
            ((bm_bf16_t*)wsrc)[i] = bm_f32_to_bf16(v);
        }
    }

    backend_buffer_t* b_in = backend_buffer_alloc(be, in_elems * sizeof(float));
    backend_buffer_t* b_w = backend_buffer_alloc(be, w_elems * (fp16 ? sizeof(bm_fp16_t) : sizeof(bm_bf16_t)));
    backend_buffer_t* b_bias = backend_buffer_alloc(be, (size_t)OC * sizeof(float));
    backend_buffer_t* b_ref = backend_buffer_alloc(be, out_elems * sizeof(float));
    backend_buffer_t* b_out = backend_buffer_alloc(be, out_elems * sizeof(float));
    backend_buffer_t* b_p = backend_buffer_alloc(be, 4 * sizeof(int));
    if (!b_in || !b_w || !b_bias || !b_ref || !b_out || !b_p) return 1;

    memcpy(backend_buffer_map(b_in), inp, in_elems * sizeof(float));
    backend_buffer_unmap(b_in);
    memcpy(backend_buffer_map(b_w), wsrc, w_elems * (fp16 ? sizeof(bm_fp16_t) : sizeof(bm_bf16_t)));
    backend_buffer_unmap(b_w);
    memcpy(backend_buffer_map(b_bias), bias, (size_t)OC * sizeof(float));
    backend_buffer_unmap(b_bias);
    int p[4] = {BT, C, OC, 0};
    memcpy(backend_buffer_map(b_p), p, sizeof(p));
    backend_buffer_unmap(b_p);

    backend_buffer_t* refbufs[] = {b_in, b_w, b_bias, b_ref, b_p};
    backend_buffer_t* outbufs[] = {b_in, b_w, b_bias, b_out, b_p};

    backend_kernel_dispatch(be, ref, refbufs, NULL, 5, OC, BT, 1, 32, 1, 1);
    int tiles_x = (OC + 31) / 32;
    int tiles_y = (BT + 31) / 32;
    backend_kernel_dispatch(be, tiled, outbufs, NULL, 5, tiles_x * 256, tiles_y, 1, 256, 1, 1);

    memcpy(ref_out, backend_buffer_map(b_ref), out_elems * sizeof(float));
    memcpy(tiled_out, backend_buffer_map(b_out), out_elems * sizeof(float));
    backend_buffer_unmap(b_ref);
    backend_buffer_unmap(b_out);

    double cos = cosine(ref_out, tiled_out, out_elems);
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_elems; i++) {
        float d = fabsf(ref_out[i] - tiled_out[i]);
        if (d > max_diff) max_diff = d;
    }
    int ok = cos > 0.999 && max_diff < 1.0f;

    double ref_ms = 0.0, til_ms = 0.0;
    if ((double)BT * C * OC > 1e6) {
        int reps = 3;
        double t0 = now_ms();
        for (int r = 0; r < reps; r++)
            backend_kernel_dispatch(be, ref, refbufs, NULL, 5, OC, BT, 1, 32, 1, 1);
        ref_ms = (now_ms() - t0) / reps;
        t0 = now_ms();
        for (int r = 0; r < reps; r++)
            backend_kernel_dispatch(be, tiled, outbufs, NULL, 5, tiles_x * 256, tiles_y, 1, 256, 1, 1);
        til_ms = (now_ms() - t0) / reps;
    }
    double gf = 2.0 * BT * C * OC / 1e9;
    printf("%s BT=%3d C=%4d OC=%4d cos=%.7f maxdiff=%.4g ref=%.2fms tiled=%.2fms x%.2f (%.0f GFLOP/s)\n",
           ok ? "PASS" : "FAIL", BT, C, OC, cos, max_diff, ref_ms, til_ms,
           til_ms > 0 ? ref_ms / til_ms : 0.0, til_ms > 0 ? gf / (til_ms / 1000.0) : 0.0);

    backend_buffer_free(b_in); backend_buffer_free(b_w); backend_buffer_free(b_bias);
    backend_buffer_free(b_ref); backend_buffer_free(b_out); backend_buffer_free(b_p);
    free(inp); free(bias); free(ref_out); free(tiled_out); free(wsrc);
    return ok ? 0 : 1;
}

int main(void) {
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    backend_kernel_t* ref_bf16 = backend_kernel_create(be, "matmul_forward_bf16");
    backend_kernel_t* til_bf16 = backend_kernel_create(be, "matmul_forward_bf16_tiled");
    backend_kernel_t* ref_fp16 = backend_kernel_create(be, "matmul_forward_fp16");
    backend_kernel_t* til_fp16 = backend_kernel_create(be, "matmul_forward_fp16_tiled");
    int fail = 0, tested = 0;
    int shapes[][3] = {
        {1, 576, 576}, {8, 576, 576}, {33, 576, 192}, {64, 576, 1536},
        {128, 576, 576}, {128, 2048, 2048}, {7, 100, 70},
    };
    if (ref_bf16 && til_bf16) {
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
            fail |= run_case(be, ref_bf16, til_bf16, shapes[i][0], shapes[i][1], shapes[i][2], 0);
        tested = 1;
    } else {
        printf("skip bf16 (kernel unavailable)\n");
    }
    if (ref_fp16 && til_fp16) {
        fail |= run_case(be, ref_fp16, til_fp16, 32, 576, 576, 1);
        fail |= run_case(be, ref_fp16, til_fp16, 128, 2048, 512, 1);
        tested = 1;
    } else {
        printf("skip fp16 (kernel unavailable)\n");
    }
    if (!tested) {
        fprintf(stderr, "no tiled kernels available\n");
        return 1;
    }

    backend_kernel_destroy(ref_bf16); backend_kernel_destroy(til_bf16);
    backend_kernel_destroy(ref_fp16); backend_kernel_destroy(til_fp16);
    backend_destroy(be);
    printf("\n=== matmul prefill tiled: %s ===\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
