// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_CHECKPOINT_H
#define BMT_CHECKPOINT_H

#include "model.h"

#define BMT_CHECKPOINT_MAGIC  20250718
#define BMT_CHECKPOINT_VERSION 1

int bmt_checkpoint_load(bm_model_t* model, const char* path);
int bmt_checkpoint_save(bm_model_t* model, const char* path);
int bmt_checkpoint_load_safetensors(bm_model_t* model, const char* dir_path);

#endif
