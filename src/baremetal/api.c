#include "baremetal.h"
#include "baremetal/context.h"
#include "baremetal/model.h"
#include "baremetal/checkpoint.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

bm_context_t* bm_create(bm_device_t device) {
    return bmt_context_create(device);
}

void bm_destroy(bm_context_t* ctx) {
    bmt_context_destroy(ctx);
}

bm_model_t* bm_create_model(bm_context_t* ctx, const bm_arch_t* arch) {
    (void)ctx;
    bm_model_t* model = calloc(1, sizeof(bm_model_t));
    if (!model) return NULL;
    if (bmt_model_alloc_buffers(model, arch) != 0) {
        free(model);
        return NULL;
    }
    return model;
}

void bm_load_weights(bm_model_t* model, const char* path) {
    if (bmt_checkpoint_load(model, path) == 0) return;
    BMT_LOG_INFO("Trying legacy llama2.c format...");
    if (bmt_checkpoint_load_legacy_llama2c(model, path) == 0) return;
    BMT_LOG_ERROR("Failed to load weights from %s", path);
}

void bm_save_weights(bm_model_t* model, const char* path) {
    bmt_checkpoint_save(model, path);
}

void bm_destroy_model(bm_model_t* model) {
    if (!model) return;
    bmt_model_free_buffers(model);
    free(model);
}

void bm_compile(bm_model_t* model) { (void)model; }
void bm_autotune(bm_model_t* model, int warmup, int iterations) { (void)model; (void)warmup; (void)iterations; }

bm_session_t* bm_create_session(bm_context_t* ctx, bm_model_t* model) { (void)ctx; (void)model; return NULL; }
float* bm_forward(bm_session_t* sess, const int* tokens, int n_tokens) { (void)sess; (void)tokens; (void)n_tokens; return NULL; }
int bm_step(bm_session_t* sess, int token) { (void)sess; (void)token; return 0; }
void bm_reset_session(bm_session_t* sess) { (void)sess; }
void bm_destroy_session(bm_session_t* sess) { free(sess); }

bm_sampler_t* bm_create_sampler(int vocab_size, float temp, float topp, uint64_t seed) { (void)vocab_size; (void)temp; (void)topp; (void)seed; return NULL; }
int bm_sample(bm_sampler_t* sampler, const float* logits) { (void)sampler; (void)logits; return 0; }
void bm_destroy_sampler(bm_sampler_t* sampler) { free(sampler); }

bm_tokenizer_t* bm_create_tokenizer(const char* path) { (void)path; return NULL; }
int* bm_encode(bm_tokenizer_t* tok, const char* text, int* n_tokens) { (void)tok; (void)text; (void)n_tokens; return NULL; }
char* bm_decode(bm_tokenizer_t* tok, int prev_token, int token) { (void)tok; (void)prev_token; (void)token; return NULL; }
void bm_destroy_tokenizer(bm_tokenizer_t* tok) { free(tok); }

int bm_get_num_parameters(const bm_model_t* model) {
    return model ? model->n_parameters : 0;
}

const char* bm_get_arch_name(const bm_model_t* model) {
    if (!model) return "none";
    const bm_arch_t* a = &model->arch;
    if (a->norm == BM_NORM_RMSNORM && a->activation == BM_ACT_SWIGLU && a->pos_enc == BM_POS_ROPE)
        return a->attention == BM_ATTN_GQA ? "Llama3-GQA" : "Llama2-MQA";
    if (a->norm == BM_NORM_LAYERNORM && a->activation == BM_ACT_GELU && a->pos_enc == BM_POS_LEARNED)
        return "GPT-2";
    return "custom";
}

void bm_print_model_info(const bm_model_t* model) {
    if (!model) return;
    const bm_arch_t* a = &model->arch;
    printf("=== %s Model ===\n", bm_get_arch_name(model));
    printf("  dim:         %d\n", a->dim);
    printf("  hidden_dim:  %d\n", a->hidden_dim);
    printf("  n_layers:    %d\n", a->n_layers);
    printf("  n_heads:     %d\n", a->n_heads);
    printf("  n_kv_heads:  %d\n", a->n_kv_heads);
    printf("  vocab_size:  %d\n", a->vocab_size);
    printf("  max_seq_len: %d\n", a->max_seq_len);
    printf("  norm:        %s\n", a->norm == BM_NORM_LAYERNORM ? "LayerNorm" : "RMSNorm");
    printf("  activation:  %s\n", a->activation == BM_ACT_GELU ? "GELU" : "SwiGLU");
    printf("  pos_enc:     %s\n", a->pos_enc == BM_POS_LEARNED ? "Learned" : "RoPE");
    printf("  attention:   %s\n", a->attention == BM_ATTN_MHA ? "MHA" : "GQA");
    printf("  bias:        %s\n", a->bias ? "yes" : "no");
    printf("  weight_tie:  %s\n", a->weight_tie ? "yes" : "no");
    printf("  parameters:  %d\n", model->n_parameters);
}
