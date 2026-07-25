#include "kernels/registry.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>

bmk_registry_t* bmk_registry_create(backend_ctx_t* ctx) {
    bmk_registry_t* reg = (bmk_registry_t*)calloc(1, sizeof(bmk_registry_t));
    if (!reg) return NULL;
    reg->ctx = ctx;
    for (int i = 0; i < BMK_MAX_OPS; i++) {
        reg->ops[i].op = (bmk_op_type_t)i;
        reg->ops[i].n_entries = 0;
    }
    return reg;
}

void bmk_registry_destroy(bmk_registry_t* reg) {
    if (!reg) return;
    for (int i = 0; i < BMK_MAX_OPS; i++) {
        for (int j = 0; j < reg->ops[i].n_entries; j++) {
            if (reg->ops[i].entries[j].kernel) {
                backend_kernel_destroy(reg->ops[i].entries[j].kernel);
            }
        }
    }
    free(reg);
}

int bmk_register(bmk_registry_t* reg, bmk_op_type_t op,
                 bmk_variant_t variant, const char* name) {
    if (!reg || op < 0 || op >= BMK_OP_COUNT) return -1;
    bmk_op_registry_t* opr = &reg->ops[op];
    if (opr->n_entries >= BMK_MAX_VARIANTS) return -1;

    backend_kernel_t* k = backend_kernel_create(reg->ctx, name);
    if (!k) {
        BMT_LOG_WARN("registry: failed to create kernel '%s'", name);
        return -1;
    }

    bmk_entry_t* e = &opr->entries[opr->n_entries++];
    e->name = name;
    e->variant = variant;
    e->kernel = k;
    return 0;
}

backend_kernel_t* bmk_select(bmk_registry_t* reg, bmk_op_type_t op,
                             int batch, int in_dim, int out_dim) {
    (void)batch;
    if (!reg || op < 0 || op >= BMK_OP_COUNT) return NULL;
    bmk_op_registry_t* opr = &reg->ops[op];
    if (opr->n_entries == 0) return NULL;

    if (op == BMK_OP_MATMUL) {
        for (int i = 0; i < opr->n_entries; i++) {
            if (opr->entries[i].variant == BMK_VARIANT_TILED &&
                (out_dim & 31) == 0 && (in_dim & 31) == 0) {
                return opr->entries[i].kernel;
            }
        }
    }

    if (op == BMK_OP_NORM_RMS || op == BMK_OP_NORM_LAYER) {
        for (int i = 0; i < opr->n_entries; i++) {
            if (opr->entries[i].variant == BMK_VARIANT_TILED &&
                in_dim >= 256) {
                return opr->entries[i].kernel;
            }
        }
    }

    if (op == BMK_OP_ATTENTION) {
        for (int i = 0; i < opr->n_entries; i++) {
            if (opr->entries[i].variant == BMK_VARIANT_FLASH) {
                return opr->entries[i].kernel;
            }
        }
    }

    for (int i = 0; i < opr->n_entries; i++) {
        if (opr->entries[i].variant == BMK_VARIANT_NAIVE) {
            return opr->entries[i].kernel;
        }
    }
    return opr->entries[0].kernel;
}