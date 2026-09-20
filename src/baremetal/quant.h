// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_QUANT_H
#define BMT_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* Q8_0 block: one fp32 scale per 32 int8 quantized values.
 * 36 bytes per 32 weights (vs 128 bytes fp32) -> ~3.55x smaller.
 * Matches the llama2.c runq.c scheme (llama2.c uses fp16 scale; we use
 * fp32 for CPU-side simplicity and native Metal float handling). */
#define Q8_BLOCK_SIZE 32

typedef struct {
    float  scale;
    int8_t q[Q8_BLOCK_SIZE];
} q8_block_t;  /* 36 bytes, naturally aligned, no padding */

/* Quantize n floats into q8 blocks. n MUST be a multiple of Q8_BLOCK_SIZE.
 * Returns the number of blocks written (n / Q8_BLOCK_SIZE), or 0 on bad input. */
size_t bmt_quantize_q8(const float* src, q8_block_t* dst, size_t n);

/* Byte size of the q8 representation for n elements (n must be a multiple of 32). */
static inline size_t bmt_q8_bytes(size_t n) {
    return (n / Q8_BLOCK_SIZE) * sizeof(q8_block_t);
}

/* Q4_0-style block: one fp32 scale per 32 int4 values, packed two per byte.
 * 20 bytes per 32 weights (vs 128 bytes fp32) -> ~6.4x smaller.
 * Values are symmetric int4 in [-8, 7]; scale = amax / 7. */
#define Q4_BLOCK_SIZE 32

typedef struct {
    float   scale;
    uint8_t qs[Q4_BLOCK_SIZE / 2];
} q4_block_t;  /* 20 bytes, naturally aligned, no padding */

/* Quantize n floats into q4 blocks. n MUST be a multiple of Q4_BLOCK_SIZE.
 * Returns the number of blocks written (n / Q4_BLOCK_SIZE), or 0 on bad input. */
size_t bmt_quantize_q4(const float* src, q4_block_t* dst, size_t n);

static inline size_t bmt_q4_bytes(size_t n) {
    return (n / Q4_BLOCK_SIZE) * sizeof(q4_block_t);
}

#endif /* BMT_QUANT_H */
