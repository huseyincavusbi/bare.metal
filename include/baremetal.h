#ifndef BAREMETAL_H
#define BAREMETAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ---- Architecture descriptor ---- */

typedef enum {
    BM_NORM_LAYERNORM,
    BM_NORM_RMSNORM,
} bm_norm_t;

typedef enum {
    BM_ACT_GELU,
    BM_ACT_SWIGLU,
} bm_act_t;

typedef enum {
    BM_POS_LEARNED,
    BM_POS_ROPE,
} bm_pos_t;

typedef enum {
    BM_ATTN_MHA,
    BM_ATTN_GQA,
} bm_attn_t;

typedef enum {
    BM_PRECISION_FP32,
    BM_PRECISION_FP16,
    BM_PRECISION_BF16,
} bm_precision_t;

typedef enum {
    BM_DEVICE_CPU,
    BM_DEVICE_METAL,
} bm_device_t;

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int padded_vocab_size;
    int max_seq_len;

    bm_norm_t      norm;
    bm_act_t       activation;
    bm_pos_t       pos_enc;
    bm_attn_t      attention;
    bm_precision_t precision;

    int bias;
    int weight_tie;

    float rope_theta;
} bm_arch_t;

/* ---- Training config ---- */

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float grad_clip;
    int   grad_accum_steps;
    int   warmup_steps;
    int   max_steps;
    int   use_master_weights;
} bm_train_config_t;

/* ---- Opaque types ---- */

typedef struct bm_context_s      bm_context_t;
typedef struct bm_model_s        bm_model_t;
typedef struct bm_session_s      bm_session_t;
typedef struct bm_trainer_s      bm_trainer_t;
typedef struct bm_sampler_s      bm_sampler_t;
typedef struct bm_tokenizer_s    bm_tokenizer_t;

/* ---- Lifecycle ---- */

bm_context_t*  bm_create(bm_device_t device);
void           bm_destroy(bm_context_t* ctx);

/* ---- Model ---- */

bm_model_t*    bm_create_model(bm_context_t* ctx, const bm_arch_t* arch);
void           bm_load_weights(bm_model_t* model, const char* path);
void           bm_save_weights(bm_model_t* model, const char* path);
void           bm_destroy_model(bm_model_t* model);

/* ---- Optimization ---- */

void           bm_compile(bm_model_t* model);
void           bm_autotune(bm_model_t* model, int warmup, int iterations);

/* ---- Inference ---- */

bm_session_t*  bm_create_session(bm_context_t* ctx, bm_model_t* model);
float*         bm_forward(bm_session_t* sess, const int* tokens, int n_tokens);
int            bm_step(bm_session_t* sess, int token);
void           bm_reset_session(bm_session_t* sess);
void           bm_destroy_session(bm_session_t* sess);

/* ---- Sampler ---- */

bm_sampler_t*  bm_create_sampler(int vocab_size, float temp, float topp,
                                 uint64_t seed);
int            bm_sample(bm_sampler_t* sampler, const float* logits);
void           bm_destroy_sampler(bm_sampler_t* sampler);

/* ---- Tokenizer ---- */

bm_tokenizer_t* bm_create_tokenizer(const char* path);
int*            bm_encode(bm_tokenizer_t* tok, const char* text, int* n_tokens);
char*           bm_decode(bm_tokenizer_t* tok, int prev_token, int token);
void            bm_destroy_tokenizer(bm_tokenizer_t* tok);

/* ---- Training ---- */

#ifdef BAREMETAL_TRAIN

bm_trainer_t*  bm_create_trainer(bm_context_t* ctx, bm_model_t* model,
                                 const bm_train_config_t* cfg);
float          bm_train_step(bm_trainer_t* trainer, const int* inputs,
                             const int* targets, int B, int T);
void           bm_load_state(bm_trainer_t* trainer, const char* path);
void           bm_save_state(bm_trainer_t* trainer, const char* path);
void           bm_destroy_trainer(bm_trainer_t* trainer);

#endif /* BAREMETAL_TRAIN */

/* ---- Utility ---- */

int            bm_get_num_parameters(const bm_model_t* model);
const char*    bm_get_arch_name(const bm_model_t* model);
void           bm_print_model_info(const bm_model_t* model);

#ifdef __cplusplus
}
#endif

#endif /* BAREMETAL_H */
