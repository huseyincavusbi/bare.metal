// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_JSON_H
#define BMT_JSON_H

#include <stddef.h>

typedef enum {
    BMT_JSON_NULL,
    BMT_JSON_BOOL,
    BMT_JSON_NUMBER,
    BMT_JSON_STRING,
    BMT_JSON_ARRAY,
    BMT_JSON_OBJECT,
} bmt_json_type_t;

typedef struct bmt_json_s bmt_json_t;

bmt_json_t*       bmt_json_parse(const char* text, char* err, size_t err_len);
void              bmt_json_free(bmt_json_t* value);

bmt_json_type_t   bmt_json_type(const bmt_json_t* value);
const bmt_json_t* bmt_json_get(const bmt_json_t* object, const char* key);
int               bmt_json_int(const bmt_json_t* value, int* out);
int               bmt_json_float(const bmt_json_t* value, float* out);
int               bmt_json_bool(const bmt_json_t* value, int* out);
const char*       bmt_json_string(const bmt_json_t* value);
int               bmt_json_array_count(const bmt_json_t* value);
const bmt_json_t* bmt_json_array_get(const bmt_json_t* value, int index);

#endif
