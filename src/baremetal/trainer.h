// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BMT_TRAINER_H
#define BMT_TRAINER_H

#include "baremetal.h"
#include "baremetal/scheduler.h"

/* Per-weight AdamW optimizer state. m and v are fp32, length n_params. */
typedef struct {
    void*           weight_ptr;   /* points into model->weight_buffer (fp32 master) */
    size_t          n_params;
    backend_buffer_t* m;          /* first moment (fp32) */
    backend_buffer_t* v;          /* second moment (fp32) */
    backend_buffer_t* par;       /* params scratch [N], written once at create */
    backend_buffer_t* master_w;  /* fp32 master weight (for mixed precision) */
    int             grad_tensor_id;
} bmt_adamw_state_t;

struct bm_trainer_s {
    bm_context_t*       ctx;
    bm_model_t*         model;
    bmt_scheduler_t*    sched;
    bmk_registry_t*     reg;
    bm_precision_t      precision;

    bmt_adamw_state_t*  opt_states;   /* one per weight tensor */
    int                 n_opt_states;

    backend_kernel_t*   k_adamw;      /* adamw_step */
    backend_buffer_t*   k_adamw_hp;   /* hyperparams scratch: [lr,b1,b2,eps,wd,bias1,bias2] */
    backend_kernel_t*   k_embed_bwd;  /* embedding_backward kernel */
    backend_buffer_t*   k_embed_param;/* [D] for embedding_backward */

    int                 step;         /* optimizer step count (for bias correction) */
    bm_train_config_t   cfg;

    float*              x_in;         /* S*D embedding scratch */
    int                 S;            /* sequence length the trainer was built for */
    int                 t_x_id;
    int                 t_logits_id;
};

/* Internal: built under BAREMETAL_TRAIN by api.c bm_create_trainer. */
bm_trainer_t* bmt_trainer_create(bm_context_t* ctx, bm_model_t* model,
                                 const bm_train_config_t* cfg, int S);

#endif /* BMT_TRAINER_H */
