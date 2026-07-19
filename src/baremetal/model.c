#include "baremetal/model.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>

int bmt_model_alloc_buffers(bm_model_t* model, const bm_arch_t* arch) {
    memset(model, 0, sizeof(bm_model_t));
    model->arch = *arch;

    model->head_size = arch->dim / arch->n_heads;
    model->kv_dim    = model->head_size * arch->n_kv_heads;
    model->n_kv_heads = arch->n_kv_heads;
    model->kv_mul     = arch->n_heads / arch->n_kv_heads;

    size_t total_bytes = 0;
    size_t vocab_size  = arch->padded_vocab_size > 0
                         ? arch->padded_vocab_size
                         : arch->vocab_size;
    int L = arch->n_layers;
    int D = arch->dim;
    int H = arch->hidden_dim;
    int HD = model->head_size;
    int NH = arch->n_heads;

    total_bytes += vocab_size * D              * sizeof(float);
    if (arch->pos_enc == BM_POS_LEARNED)
        total_bytes += arch->max_seq_len * D   * sizeof(float);
    if (arch->norm == BM_NORM_LAYERNORM) {
        total_bytes += L * D                   * sizeof(float);
        total_bytes += L * D                   * sizeof(float);
        total_bytes += L * D                   * sizeof(float);
        total_bytes += L * D                   * sizeof(float);
    } else {
        total_bytes += L * D                   * sizeof(float);
        total_bytes += L * D                   * sizeof(float);
    }
    int NKV = arch->n_kv_heads;
    total_bytes += L * NH * HD * D             * sizeof(float);
    total_bytes += L * NKV * HD * D            * sizeof(float);
    total_bytes += L * NKV * HD * D            * sizeof(float);
    if (arch->bias) total_bytes += L * 3 * D   * sizeof(float);
    if (arch->has_qk_norm) {
        total_bytes += L * NH * HD             * sizeof(float);
        total_bytes += L * arch->n_kv_heads * HD * sizeof(float);
    }
    total_bytes += L * NH * HD * D             * sizeof(float);
    if (arch->bias) total_bytes += L * D       * sizeof(float);
    total_bytes += L * H * D                   * sizeof(float);
    if (arch->bias) total_bytes += L * H       * sizeof(float);
    if (arch->activation == BM_ACT_SWIGLU)
        total_bytes += L * H * D               * sizeof(float);
    total_bytes += L * D * H                   * sizeof(float);
    if (arch->bias) total_bytes += L * D       * sizeof(float);
    if (arch->has_ffn_post_norm) {
        total_bytes += L * D * 2              * sizeof(float);
    }
    if (arch->norm == BM_NORM_LAYERNORM)
        total_bytes += D                       * sizeof(float);
    if (!arch->weight_tie)
        total_bytes += vocab_size * D          * sizeof(float);

    model->n_parameters = total_bytes / sizeof(float);

    void* raw = calloc(1, total_bytes);
    if (!raw) {
        BMT_LOG_ERROR("Failed to allocate weight buffer (%zu bytes)", total_bytes);
        return -1;
    }
    float* w = (float*)raw;

    model->token_embedding_table = w; w += vocab_size * D;
    if (arch->pos_enc == BM_POS_LEARNED) {
        model->wpe = w; w += arch->max_seq_len * D;
    }

    if (arch->norm == BM_NORM_LAYERNORM) {
        model->ln1w = w; w += L * D;
        model->ln1b = w; w += L * D;
    } else {
        model->ln1w = w; w += L * D;
    }
    model->qw = w; w += L * NH * HD * D;
    model->kw = w; w += L * NKV * HD * D;
    model->vw = w; w += L * NKV * HD * D;
    if (arch->bias) { model->qkvb = w; w += L * (NH + 2*NKV) * HD; }
    if (arch->has_qk_norm) {
        model->q_norm_w = w; w += L * NH * HD;
        model->k_norm_w = w; w += L * arch->n_kv_heads * HD;
    }
    model->attprojw = w; w += L * NH * HD * D;
    if (arch->bias) { model->attprojb = w; w += L * D; }

    if (arch->norm == BM_NORM_LAYERNORM) {
        model->ln2w = w; w += L * D;
        model->ln2b = w; w += L * D;
    } else {
        model->ln2w = w; w += L * D;
    }

    model->fcw = w; w += L * H * D;
    if (arch->bias) { model->fcb = w; w += L * H; }
    if (arch->activation == BM_ACT_SWIGLU) { model->fcw3 = w; w += L * H * D; }
    model->fcprojw = w; w += L * D * H;
    if (arch->bias) { model->fcprojb = w; w += L * D; }

    if (arch->has_ffn_post_norm) {
        model->pre_ffn_w = w; w += L * D;
        model->ffn_post_w = w; w += L * D;
    }

    model->lnfw = w; w += D;
    if (arch->norm == BM_NORM_LAYERNORM) { model->lnfb = w; w += D; }

    if (arch->weight_tie) {
        model->wcls = model->token_embedding_table;
    } else {
        model->wcls = w; w += vocab_size * D;
    }

    model->weight_buffer = (void*)raw;
    BMT_LOG_INFO("Allocated %zu bytes for %d parameters (arch=%d)",
                 total_bytes, model->n_parameters, arch->norm);
    return 0;
}

void bmt_model_free_buffers(bm_model_t* model) {
    if (!model) return;
    free(model->weight_buffer);
    memset(model, 0, sizeof(bm_model_t));
}
