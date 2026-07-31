// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/quant.h"
#include <math.h>

size_t bmt_quantize_q8(const float* src, q8_block_t* dst, size_t n) {
    if (n == 0 || (n % Q8_BLOCK_SIZE) != 0) return 0;
    size_t nblocks = n / Q8_BLOCK_SIZE;
    for (size_t b = 0; b < nblocks; b++) {
        const float* p = src + b * Q8_BLOCK_SIZE;
        float amax = 0.0f;
        for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
            float a = fabsf(p[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        dst[b].scale = scale;
        if (scale == 0.0f) {
            for (int i = 0; i < Q8_BLOCK_SIZE; i++) dst[b].q[i] = 0;
        } else {
            float inv = 1.0f / scale;
            for (int i = 0; i < Q8_BLOCK_SIZE; i++) {
                int q = (int)lroundf(p[i] * inv);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                dst[b].q[i] = (int8_t)q;
            }
        }
    }
    return nblocks;
}
