#include "baremetal/quant.h"
#include "baremetal/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static uint32_t rng_state = 7;
static float rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

static int q4_case(int n) {
    bm_bf16_t* b = malloc((size_t)n * sizeof(bm_bf16_t));
    float* f = malloc((size_t)n * sizeof(float));
    q4_block_t* qa = calloc((size_t)n / Q4_BLOCK_SIZE, sizeof(q4_block_t));
    q4_block_t* qb = calloc((size_t)n / Q4_BLOCK_SIZE, sizeof(q4_block_t));
    if (!b || !f || !qa || !qb) return 1;
    for (int i = 0; i < n; i++) {
        b[i] = bm_f32_to_bf16(rnd() * 5.0f);
        f[i] = bm_bf16_to_f32(b[i]);
    }
    size_t na = bmt_quantize_q4_bf16(b, qa, (size_t)n);
    size_t nb = bmt_quantize_q4(f, qb, (size_t)n);
    int ok = na == nb && na == (size_t)n / Q4_BLOCK_SIZE &&
             memcmp(qa, qb, na * sizeof(q4_block_t)) == 0;
    free(b); free(f); free(qa); free(qb);
    return ok ? 0 : 1;
}

static int q8_case(int n) {
    bm_bf16_t* b = malloc((size_t)n * sizeof(bm_bf16_t));
    float* f = malloc((size_t)n * sizeof(float));
    q8_block_t* qa = calloc((size_t)n / Q8_BLOCK_SIZE, sizeof(q8_block_t));
    q8_block_t* qb = calloc((size_t)n / Q8_BLOCK_SIZE, sizeof(q8_block_t));
    if (!b || !f || !qa || !qb) return 1;
    for (int i = 0; i < n; i++) {
        b[i] = bm_f32_to_bf16(rnd() * 50.0f);
        f[i] = bm_bf16_to_f32(b[i]);
    }
    size_t na = bmt_quantize_q8_bf16(b, qa, (size_t)n);
    size_t nb = bmt_quantize_q8(f, qb, (size_t)n);
    int ok = na == nb && na == (size_t)n / Q8_BLOCK_SIZE &&
             memcmp(qa, qb, na * sizeof(q8_block_t)) == 0;
    free(b); free(f); free(qa); free(qb);
    return ok ? 0 : 1;
}

static int zero_case(void) {
    int n = 128;
    bm_bf16_t b[128];
    float f[128];
    q4_block_t q4[4];
    q8_block_t q8[4];
    for (int i = 0; i < n; i++) { b[i] = bm_f32_to_bf16(0.0f); f[i] = 0.0f; }
    bmt_quantize_q4_bf16(b, q4, (size_t)n);
    bmt_quantize_q8_bf16(b, q8, (size_t)n);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (q4[i].scale != 0.0f || q8[i].scale != 0.0f) ok = 0;
        for (int j = 0; j < 16; j++) if (q4[i].qs[j] != 0) ok = 0;
        for (int j = 0; j < 32; j++) if (q8[i].q[j] != 0) ok = 0;
    }
    (void)f;
    return ok ? 0 : 1;
}

static int invalid_case(void) {
    float f[64];
    bm_bf16_t b[64];
    q4_block_t q4[2];
    q8_block_t q8[2];
    for (int i = 0; i < 64; i++) { f[i] = 1.0f; b[i] = bm_f32_to_bf16(1.0f); }
    int ok = bmt_quantize_q4(f, q4, 31) == 0 &&
             bmt_quantize_q8(f, q8, 31) == 0 &&
             bmt_quantize_q4_bf16(b, q4, 31) == 0 &&
             bmt_quantize_q8_bf16(b, q8, 31) == 0 &&
             bmt_quantize_q4(f, q4, 0) == 0 &&
             bmt_quantize_q8_bf16(b, q8, 0) == 0;
    return ok ? 0 : 1;
}

int main(void) {
    int fail = 0;
    int sizes[] = {32, 1024, 4096};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        fail |= q4_case(sizes[i]);
        fail |= q8_case(sizes[i]);
    }
    fail |= zero_case();
    fail |= invalid_case();
    printf("=== quant bf16 paths: %s ===\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
