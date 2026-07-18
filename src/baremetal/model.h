#ifndef BMT_MODEL_H
#define BMT_MODEL_H

#include "baremetal.h"

struct bm_model_s {
    bm_arch_t arch;

    int      n_parameters;

    int      kv_dim;
    int      head_size;
    int      n_kv_heads;
    int      kv_mul;

    void*    weight_buffer;

    float*   token_embedding_table;
    float*   wpe;
    float*   ln1w;
    float*   ln1b;
    float*   qkvw;
    float*   qkvb;
    float*   attprojw;
    float*   attprojb;
    float*   ln2w;
    float*   ln2b;
    float*   fcw;
    float*   fcb;
    float*   fcw3;
    float*   fcprojw;
    float*   fcprojb;
    float*   lnfw;
    float*   lnfb;
    float*   wcls;
};

int  bmt_model_alloc_buffers(bm_model_t* model, const bm_arch_t* arch);
void bmt_model_free_buffers(bm_model_t* model);
int  bmt_model_load_weights(bm_model_t* model, const char* path);

#endif
