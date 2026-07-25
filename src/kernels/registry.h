#ifndef BMT_KERNEL_REGISTRY_H
#define BMT_KERNEL_REGISTRY_H

#include "backend/backend.h"

#define BMK_MAX_VARIANTS 8
#define BMK_MAX_OPS 16

typedef enum {
    BMK_OP_MATMUL = 0,
    BMK_OP_NORM_RMS,
    BMK_OP_NORM_LAYER,
    BMK_OP_ACT_GELU,
    BMK_OP_ACT_SWIGLU,
    BMK_OP_POS_ENC_ROPE,
    BMK_OP_ATTENTION,
    BMK_OP_FUSED_RESIDUAL_NORM,
    BMK_OP_FUSED_CLASSIFIER,
    BMK_OP_COUNT
} bmk_op_type_t;

typedef enum {
    BMK_VARIANT_NAIVE = 0,
    BMK_VARIANT_TILED,
    BMK_VARIANT_FLASH,
} bmk_variant_t;

typedef struct {
    const char* name;
    bmk_variant_t variant;
    backend_kernel_t* kernel;
} bmk_entry_t;

typedef struct {
    bmk_op_type_t op;
    bmk_entry_t entries[BMK_MAX_VARIANTS];
    int n_entries;
} bmk_op_registry_t;

typedef struct {
    backend_ctx_t* ctx;
    bmk_op_registry_t ops[BMK_MAX_OPS];
} bmk_registry_t;

bmk_registry_t* bmk_registry_create(backend_ctx_t* ctx);
void bmk_registry_destroy(bmk_registry_t* reg);

int bmk_register(bmk_registry_t* reg, bmk_op_type_t op,
                 bmk_variant_t variant, const char* name);

backend_kernel_t* bmk_select(bmk_registry_t* reg, bmk_op_type_t op,
                             int batch, int in_dim, int out_dim);

#endif