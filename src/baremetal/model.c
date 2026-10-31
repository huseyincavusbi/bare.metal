// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/model.h"
#include "baremetal/graph.h"
#include "utils/log.h"
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bmt_arch_validate(const bm_arch_t* arch, int has_rope_scaling) {
    const char* names[6];
    int n = 0;

    if (arch->bias)              names[n++] = "attention_bias";
    if (arch->has_qk_norm)       names[n++] = "qk_norm";
    if (arch->has_ffn_post_norm) names[n++] = "post_ffn_norm";
    if (arch->gemma_norm)        names[n++] = "gemma_norm";
    if (arch->embed_scale)       names[n++] = "embed_scale";
    if (has_rope_scaling && arch->rope_scaling != BM_ROPE_LLAMA3)
        names[n++] = "rope_scaling";

    if (n == 0) return 0;

    char list[128];
    size_t off = 0;
    for (int i = 0; i < n && off < sizeof(list); i++) {
        int wrote = snprintf(list + off, sizeof(list) - off, "%s%s",
                             i > 0 ? ", " : "", names[i]);
        if (wrote < 0) break;
        off += (size_t)wrote;
    }

    BMT_LOG_ERROR("Unsupported architecture feature%s: %s",
                  n > 1 ? "s" : "", list);
    return -1;
}

int bmt_model_alloc_buffers(bm_model_t* model, const bm_arch_t* arch) {
    return bmt_model_alloc_buffers_ex(model, arch, 0);
}

int bmt_model_alloc_buffers_ex(bm_model_t* model, const bm_arch_t* arch, int use_mmap) {
    memset(model, 0, sizeof(bm_model_t));
    model->arch = *arch;

    model->head_size = arch->head_dim > 0 ? arch->head_dim : (arch->dim / arch->n_heads);
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
    if (arch->bias) total_bytes += L * ((int)NH + 2*(int)NKV) * (int)HD * sizeof(float);
    if (arch->has_qk_norm) {
        total_bytes += L * HD                 * sizeof(float);
        total_bytes += L * arch->n_kv_heads * HD * sizeof(float);
    }
    total_bytes += L * NH * HD * D             * sizeof(float);
    if (arch->bias) total_bytes += L * NH * HD * sizeof(float);
    total_bytes += L * H * D                   * sizeof(float);
    if (arch->bias) total_bytes += L * H       * sizeof(float);
    if (arch->gated_mlp)
        total_bytes += L * H * D               * sizeof(float);
    total_bytes += L * D * H                   * sizeof(float);
    if (arch->bias) total_bytes += L * D       * sizeof(float);
    if (arch->has_ffn_post_norm) {
        total_bytes += L * D * 2              * sizeof(float);
    }
    total_bytes += D                           * sizeof(float);
    if (arch->norm == BM_NORM_LAYERNORM)
        total_bytes += D                       * sizeof(float);
    if (!arch->weight_tie)
        total_bytes += vocab_size * D          * sizeof(float);

    model->n_parameters = total_bytes / sizeof(float);

    void* raw = NULL;
    if (use_mmap) {
        raw = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw != MAP_FAILED) {
            model->weight_is_mmap = 1;
            model->weight_map_size = total_bytes;
        } else {
            raw = NULL;
        }
    }
    if (!raw) {
        raw = calloc(1, total_bytes);
        if (!raw) {
            BMT_LOG_ERROR("Failed to allocate weight buffer (%zu bytes)", total_bytes);
            return -1;
        }
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
    if (arch->bias) { model->qkvb = w; w += L * ((int)NH + 2*(int)NKV) * (int)HD; }
    if (arch->has_qk_norm) {
        model->q_norm_w = w; w += L * HD;
        model->k_norm_w = w; w += L * arch->n_kv_heads * HD;
    }
    model->attprojw = w; w += L * NH * HD * D;
    if (arch->bias) { model->attprojb = w; w += L * NH * HD; }

    if (arch->norm == BM_NORM_LAYERNORM) {
        model->ln2w = w; w += L * D;
        model->ln2b = w; w += L * D;
    } else {
        model->ln2w = w; w += L * D;
    }

    model->fcw = w; w += L * H * D;
    if (arch->bias) { model->fcb = w; w += L * H; }
    if (arch->gated_mlp) { model->fcw3 = w; w += L * H * D; }
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
    BMT_LOG_INFO("Allocated %zu bytes for %zu parameters (arch=%d)",
                 total_bytes, model->n_parameters, arch->norm);
    return 0;
}

void bmt_model_free_buffers(bm_model_t* model) {
    if (!model) return;
    if (model->stream_close && model->stream_set) {
        model->stream_close(model->stream_set);
        model->stream_close = NULL;
        model->stream_set = NULL;
    }
    if (model->weight_is_mmap && model->weight_buffer)
        munmap(model->weight_buffer, model->weight_map_size);
    else
        free(model->weight_buffer);
    free(model->weight_srcs);
    memset(model, 0, sizeof(bm_model_t));
}

void bmt_model_release_host_weights(bm_model_t* model) {
    if (!model || !model->weight_buffer) return;
    size_t vocab_size = model->arch.padded_vocab_size > 0
                        ? model->arch.padded_vocab_size
                        : model->arch.vocab_size;
    size_t emb_bytes = vocab_size * (size_t)model->arch.dim * sizeof(float);
    size_t wpe_bytes = (model->arch.pos_enc == BM_POS_LEARNED)
                       ? (size_t)model->arch.max_seq_len * (size_t)model->arch.dim * sizeof(float)
                       : 0;
    size_t keep = emb_bytes + wpe_bytes;
    if (model->weight_is_mmap) {
        if (keep < model->weight_map_size)
            madvise((char*)model->weight_buffer + keep, model->weight_map_size - keep, MADV_DONTNEED);
        model->token_embedding_table = (float*)model->weight_buffer;
        model->wpe = wpe_bytes ? (float*)((char*)model->weight_buffer + emb_bytes) : NULL;
    } else {
        void* smaller = realloc(model->weight_buffer, keep);
        if (!smaller) return;
        model->weight_buffer = smaller;
        model->token_embedding_table = (float*)smaller;
        model->wpe = wpe_bytes ? (float*)((char*)smaller + emb_bytes) : NULL;
    }
    bmt_graph_t* graph = (bmt_graph_t*)model->graph;
    if (graph) {
        for (int i = 0; i < graph->n_tensors; i++) {
            graph->tensors[i].weight_ptr = NULL;
            graph->tensors[i].src_ptr = NULL;
            graph->tensors[i].src_is_bf16 = 0;
        }
    }
    free(model->weight_srcs);
    model->weight_srcs = NULL;
    model->n_weight_srcs = 0;
    if (model->stream_close && model->stream_set) {
        model->stream_close(model->stream_set);
        model->stream_close = NULL;
        model->stream_set = NULL;
    }
}
