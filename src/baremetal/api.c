#include "baremetal.h"
#include "baremetal/context.h"
#include "baremetal/model.h"
#include "baremetal/checkpoint.h"
#include "baremetal/sampler.h"
#include "baremetal/tokenizer.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
    // Try bare.metal format first
    if (bmt_checkpoint_load(model, path) == 0) return;
    // Try legacy llama2.c format
    BMT_LOG_INFO("Trying legacy llama2.c format...");
    if (bmt_checkpoint_load_legacy_llama2c(model, path) == 0) return;
    // Try directory (safetensors)
    if (bmt_checkpoint_load_safetensors(model, path) == 0) return;
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

#include "baremetal/graph.h"
#include "baremetal/compiler.h"

void bm_compile(bm_model_t* model) {
    if (!model) return;
    bmt_graph_build(model);
    bmt_compiler_run(model->graph);
}

/* Kernel/variant autotuning is not yet implemented; the registry's
 * bmk_select() already picks the best *available* variant heuristically
 * (tiled matmul when dims align, warp-reduced norms, flash attention).
 * This is an explicit no-op until a real timing-based autotuner exists. */
void bm_autotune(bm_model_t* model, int warmup, int iterations) { (void)model; (void)warmup; (void)iterations; }

/* ---- Sampler ---- */

bm_sampler_t* bm_create_sampler(int vocab_size, float temp, float topp, uint64_t seed) {
    bm_sampler_t* s = (bm_sampler_t*)calloc(1, sizeof(bm_sampler_t));
    if (!s) return NULL;
    s->vocab_size = vocab_size;
    s->temperature = temp;
    s->topp = topp;
    s->top_k = 40;             /* matches the validated run-path default */
    s->rng_state = (unsigned int)seed;
    if (seed == 0) s->rng_state = (unsigned int)(uintptr_t)s;  /* non-zero default */
    s->probs = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (!s->probs) { free(s); return NULL; }
    return s;
}

void bm_destroy_sampler(bm_sampler_t* sampler) {
    if (!sampler) return;
    free(sampler->probs);
    free(sampler);
}

/* ---- Tokenizer ---- */

bm_tokenizer_t* bm_create_tokenizer(const char* path) {
    if (!path) return NULL;
    /* vocab_size is read from the model checkpoint; the public API takes only
     * a path, so we open config.json to get vocab_size before init. */
    int vocab_size = 0;
    char cfg[1024];
    snprintf(cfg, sizeof(cfg), "%s/config.json", path);
    FILE* f = fopen(cfg, "rb");
    if (f) {
        char buf[1 << 16];
        size_t rd = fread(buf, 1, sizeof(buf) - 1, f);
        buf[rd] = '\0';
        fclose(f);
        const char* p = strstr(buf, "\"vocab_size\"");
        if (p) { p = strchr(p, ':'); if (p) vocab_size = atoi(p + 1); }
    }
    if (vocab_size <= 0) { BMT_LOG_ERROR("bm_create_tokenizer: no vocab_size in %s", cfg); return NULL; }

    bm_tokenizer_t* tok = (bm_tokenizer_t*)calloc(1, sizeof(bm_tokenizer_t));
    if (!tok) return NULL;
    bm_tokenizer_init(tok, path, vocab_size);
    return tok;
}

int* bm_encode(bm_tokenizer_t* tok, const char* text, int* n_tokens) {
    if (!tok || !text || !n_tokens) return NULL;
    int cap = (int)strlen(text) + 8;          /* worst case ~1 token/char + BOS/EOS */
    int* tokens = (int*)malloc((size_t)cap * sizeof(int));
    if (!tokens) return NULL;
    int n = 0;
    if (bm_tokenizer_encode(tok, text, 0, 0, tokens, &n) != 0) {
        free(tokens);
        return NULL;
    }
    *n_tokens = n;
    return tokens;
}

char* bm_decode(bm_tokenizer_t* tok, int prev_token, int token) {
    if (!tok) return NULL;
    return bm_tokenizer_decode(tok, prev_token, token);
}

void bm_destroy_tokenizer(bm_tokenizer_t* tok) {
    if (!tok) return;
    bm_tokenizer_free(tok);
    free(tok);
}

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
    printf("  parameters:  %zu\n", model->n_parameters);
}

#ifdef BAREMETAL_TRAIN
#include "baremetal/trainer.h"

bm_trainer_t* bm_create_trainer(bm_context_t* ctx, bm_model_t* model,
                                const bm_train_config_t* cfg) {
    int S = (cfg->seq_len > 0) ? cfg->seq_len : 64;
    return bmt_trainer_create(ctx, model, cfg, S);
}
#endif /* BAREMETAL_TRAIN */
