#include "baremetal.h"
#include "baremetal/checkpoint.h"
#include "baremetal/model.h"
#include <stdio.h>
#include <string.h>

static void print_arch(const char* label, const bm_model_t* m) {
    const bm_arch_t* a = &m->arch;
    printf("=== %s ===\n", label);
    printf("  dim:           %d\n", a->dim);
    printf("  hidden_dim:    %d\n", a->hidden_dim);
    printf("  n_layers:      %d\n", a->n_layers);
    printf("  n_heads:       %d\n", a->n_heads);
    printf("  n_kv_heads:    %d\n", a->n_kv_heads);
    printf("  vocab_size:    %d\n", a->vocab_size);
    printf("  padded_vocab:  %d\n", a->padded_vocab_size);
    printf("  max_seq_len:   %d\n", a->max_seq_len);
    printf("  norm:          %s\n", a->norm == BM_NORM_LAYERNORM ? "LN" : "RMS");
    printf("  activation:    %s\n", a->activation == BM_ACT_GELU ? "GELU" : "SwiGLU");
    printf("  pos_enc:       %s\n", a->pos_enc == BM_POS_LEARNED ? "Learned" : "RoPE");
    printf("  attention:     %s\n", a->attention == BM_ATTN_MHA ? "MHA" : "GQA");
    printf("  bias:          %d\n", a->bias);
    printf("  weight_tie:    %d\n", a->weight_tie);
    printf("  has_qk_norm:   %d\n", a->has_qk_norm);
    printf("  has_ffn_post:  %d\n", a->has_ffn_post_norm);
    printf("  head_dim:      %d (head_size=%d)\n", a->head_dim, m->head_size);
    printf("  gated_mlp:     %d\n", a->gated_mlp);
    printf("  gemma_norm:    %d\n", a->gemma_norm);
    printf("  embed_scale:   %d\n", a->embed_scale);
    printf("  weight_layout: %d\n", a->weight_layout);
    printf("  rope_theta:    %f\n", a->rope_theta);
    printf("  n_parameters:  %d\n", m->n_parameters);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bin_path> <st_dir>\n", argv[0]);
        return 1;
    }
    bm_model_t m1 = {0}, m2 = {0};

    if (bmt_checkpoint_load(&m1, argv[1]) != 0) {
        fprintf(stderr, "binary load failed\n"); return 1;
    }
    print_arch("BINARY", &m1);

    if (bmt_checkpoint_load_safetensors(&m2, argv[2]) != 0) {
        fprintf(stderr, "st load failed\n"); return 1;
    }
    print_arch("SAFETENSORS", &m2);

    return 0;
}
