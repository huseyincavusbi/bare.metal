// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_CONTEXT_H
#define BMT_CONTEXT_H

#include "backend/backend.h"

struct bm_context_s {
    bm_device_t    device_type;
    backend_ctx_t* backend_ctx;
};

bm_context_t* bmt_context_create(bm_device_t device);
void          bmt_context_destroy(bm_context_t* ctx);

#endif
