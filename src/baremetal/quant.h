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

#endif /* BMT_QUANT_H */
