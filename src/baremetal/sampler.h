// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_SAMPLER_H
#define BMT_SAMPLER_H

#include "baremetal.h"

/* Concrete sampler state behind the opaque bm_sampler_t* in baremetal.h.
 * Carries temperature, top-p, top-k, an LCG RNG state, and a reusable
 * probability scratch buffer (allocated once at creation, sized to vocab). */
struct bm_sampler_s {
    int          vocab_size;
    float        temperature;
    float        topp;
    int          top_k;
    unsigned int rng_state;
    float*       probs;     /* scratch, length vocab_size */
};

#endif /* BMT_SAMPLER_H */
