#ifndef BAREMETAL_TYPES_H
#define BAREMETAL_TYPES_H

#include <stdint.h>
#include <string.h>

typedef uint16_t bm_bf16_t;
typedef uint16_t bm_fp16_t;

static inline float bm_bf16_to_f32(bm_bf16_t x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline bm_bf16_t bm_f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    u += rounding_bias;
    return (bm_bf16_t)(u >> 16);
}

static inline void bm_f32_to_bf16_array(const float* src, bm_bf16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = bm_f32_to_bf16(src[i]);
}

static inline void bm_bf16_to_f32_array(const bm_bf16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = bm_bf16_to_f32(src[i]);
}

#endif
