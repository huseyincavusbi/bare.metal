#ifndef BMT_CHECKPOINT_H
#define BMT_CHECKPOINT_H

#include "model.h"

#define BMT_CHECKPOINT_MAGIC  20250718
#define BMT_CHECKPOINT_VERSION 1

int bmt_checkpoint_load(bm_model_t* model, const char* path);
int bmt_checkpoint_save(bm_model_t* model, const char* path);

int bmt_checkpoint_load_legacy_llama2c(bm_model_t* model, const char* path);

#endif
