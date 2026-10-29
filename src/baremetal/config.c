// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/config.h"
#include "baremetal/json.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const bmt_tensor_map_t k_llama_tensors = {
    .embed          = "model.embed_tokens.weight",
    .layers         = "model.layers.",
    .input_norm     = "input_layernorm.weight",
    .post_attn_norm = "post_attention_layernorm.weight",
    .q              = "self_attn.q_proj.weight",
    .k              = "self_attn.k_proj.weight",
    .v              = "self_attn.v_proj.weight",
    .o              = "self_attn.o_proj.weight",
    .gate           = "mlp.gate_proj.weight",
    .up             = "mlp.up_proj.weight",
    .down           = "mlp.down_proj.weight",
    .final_norm     = "model.norm.weight",
    .lm_head        = "lm_head.weight",
    .q_norm         = "self_attn.q_norm.weight",
    .k_norm         = "self_attn.k_norm.weight",
    .pre_ffn_norm   = "pre_feedforward_layernorm.weight",
    .post_ffn_norm  = "post_feedforward_layernorm.weight",
};

static const bmt_arch_spec_t k_specs[] = {
    { "llama", BMT_ARCH_FAMILY_LLAMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 1, NULL },
    { "smollm", BMT_ARCH_FAMILY_LLAMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 1, NULL },
    { "qwen2", BMT_ARCH_FAMILY_QWEN2, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "linear attention/MLP biases are not implemented" },
    { "qwen3", BMT_ARCH_FAMILY_QWEN3, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "qk-norm is not implemented" },
    { "gemma", BMT_ARCH_FAMILY_GEMMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC,
      BMT_WEIGHT_TRANSFORM_GEMMA_NORM | BMT_WEIGHT_TRANSFORM_EMBED_SCALE |
      BMT_WEIGHT_TRANSFORM_LOGIT_CAP, 0,
      "gemma norm/embed scaling is not implemented" },
    { "gemma2", BMT_ARCH_FAMILY_GEMMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC,
      BMT_WEIGHT_TRANSFORM_GEMMA_NORM | BMT_WEIGHT_TRANSFORM_EMBED_SCALE |
      BMT_WEIGHT_TRANSFORM_LOGIT_CAP, 0,
      "gemma norm/embed scaling is not implemented" },
    { "gemma3", BMT_ARCH_FAMILY_GEMMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC,
      BMT_WEIGHT_TRANSFORM_GEMMA_NORM | BMT_WEIGHT_TRANSFORM_EMBED_SCALE |
      BMT_WEIGHT_TRANSFORM_LOGIT_CAP, 0,
      "gemma norm/embed scaling is not implemented" },
    { "gemma3_text", BMT_ARCH_FAMILY_GEMMA, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC,
      BMT_WEIGHT_TRANSFORM_GEMMA_NORM | BMT_WEIGHT_TRANSFORM_EMBED_SCALE |
      BMT_WEIGHT_TRANSFORM_LOGIT_CAP, 0,
      "gemma norm/embed scaling is not implemented" },
    { "gpt2", BMT_ARCH_FAMILY_GPT2, NULL,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "GPT-2 tensor layout and linear biases are not implemented" },
    { "phi3", BMT_ARCH_FAMILY_PHI3, NULL,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "fused gate/up projections are not implemented" },
    { "mistral", BMT_ARCH_FAMILY_MISTRAL, &k_llama_tensors,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "sliding-window attention is not implemented" },
    { "mixtral", BMT_ARCH_FAMILY_MIXTRAL, NULL,
      BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
      "MoE is not implemented" },
};

static const bmt_arch_spec_t k_unknown_spec = {
    "unknown", BMT_ARCH_FAMILY_UNKNOWN, NULL,
    BMT_ARCH_BUILDER_GENERIC, BMT_WEIGHT_TRANSFORM_NONE, 0,
    "unrecognized model_type"
};

static char* read_text_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long len = ftell(f);
    if (len < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char* buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';
    return buf;
}

static void lowercase_copy(char* dst, size_t dst_len, const char* src) {
    size_t i = 0;
    if (!src) { dst[0] = '\0'; return; }
    for (; src[i] && i + 1 < dst_len; i++) {
        char c = src[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        dst[i] = c;
    }
    dst[i] = '\0';
}

static void copy_string(char* dst, size_t dst_len, const char* src) {
    if (!src) { dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= dst_len) n = dst_len - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static void config_int(const bmt_json_t* obj, const char* key, int* out) {
    const bmt_json_t* v = bmt_json_get(obj, key);
    if (v) bmt_json_int(v, out);
}

static void config_bool(const bmt_json_t* obj, const char* key, int* out) {
    const bmt_json_t* v = bmt_json_get(obj, key);
    if (v) bmt_json_bool(v, out);
}

int bmt_config_load(const char* dir_path, bmt_config_t* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir_path);
    char* text = read_text_file(path);
    if (!text) {
        BMT_LOG_ERROR("Cannot read %s", path);
        return -1;
    }

    char err[160];
    bmt_json_t* root = bmt_json_parse(text, err, sizeof(err));
    free(text);
    if (!root) {
        BMT_LOG_ERROR("Invalid JSON in %s: %s", path, err);
        return -1;
    }
    if (bmt_json_type(root) != BMT_JSON_OBJECT) {
        BMT_LOG_ERROR("%s: top-level JSON value must be an object", path);
        bmt_json_free(root);
        return -1;
    }

    const bmt_json_t* section = bmt_json_get(root, "text_config");
    if (bmt_json_type(section) != BMT_JSON_OBJECT) section = root;

    const char* model_type = bmt_json_string(bmt_json_get(root, "model_type"));
    if (!model_type) model_type = bmt_json_string(bmt_json_get(section, "model_type"));
    lowercase_copy(cfg->model_type, sizeof(cfg->model_type), model_type);

    const bmt_json_t* archs = bmt_json_get(root, "architectures");
    if (bmt_json_type(archs) != BMT_JSON_ARRAY) archs = bmt_json_get(section, "architectures");
    if (bmt_json_type(archs) == BMT_JSON_ARRAY && bmt_json_array_count(archs) > 0)
        copy_string(cfg->architecture, sizeof(cfg->architecture),
                    bmt_json_string(bmt_json_array_get(archs, 0)));

    config_int(section, "hidden_size", &cfg->hidden_size);
    config_int(section, "intermediate_size", &cfg->intermediate_size);
    config_int(section, "num_hidden_layers", &cfg->num_hidden_layers);
    config_int(section, "num_attention_heads", &cfg->num_attention_heads);
    config_int(section, "num_key_value_heads", &cfg->num_key_value_heads);
    config_int(section, "vocab_size", &cfg->vocab_size);
    config_int(section, "max_position_embeddings", &cfg->max_position_embeddings);
    config_int(section, "head_dim", &cfg->head_dim);

    if (cfg->hidden_size == 0) config_int(section, "n_embd", &cfg->hidden_size);
    if (cfg->num_hidden_layers == 0) config_int(section, "n_layer", &cfg->num_hidden_layers);
    if (cfg->num_attention_heads == 0) config_int(section, "n_head", &cfg->num_attention_heads);
    if (cfg->max_position_embeddings == 0) config_int(section, "n_positions", &cfg->max_position_embeddings);
    if (cfg->intermediate_size == 0) config_int(section, "n_inner", &cfg->intermediate_size);

    config_bool(section, "attention_bias", &cfg->attention_bias);
    config_bool(section, "mlp_bias", &cfg->mlp_bias);
    config_bool(section, "tie_word_embeddings", &cfg->tie_word_embeddings);

    cfg->has_rms_norm_eps = bmt_json_get(section, "rms_norm_eps") != NULL;

    const bmt_json_t* theta = bmt_json_get(section, "rope_theta");
    if (bmt_json_type(theta) == BMT_JSON_NUMBER) {
        cfg->has_rope_theta = 1;
        bmt_json_float(theta, &cfg->rope_theta);
    }

    const bmt_json_t* scaling = bmt_json_get(section, "rope_scaling");
    cfg->has_rope_scaling = bmt_json_type(scaling) == BMT_JSON_OBJECT;

    const char* act = bmt_json_string(bmt_json_get(section, "hidden_act"));
    if (!act) act = bmt_json_string(bmt_json_get(section, "hidden_activation"));
    if (!act) act = bmt_json_string(bmt_json_get(section, "activation_function"));
    copy_string(cfg->hidden_act, sizeof(cfg->hidden_act), act);

    bmt_json_free(root);
    return 0;
}

static const bmt_arch_spec_t* spec_by_name(const char* name) {
    if (!name || !name[0]) return NULL;
    for (size_t i = 0; i < sizeof(k_specs) / sizeof(k_specs[0]); i++) {
        if (strcmp(k_specs[i].name, name) == 0)
            return &k_specs[i];
    }
    return NULL;
}

static const bmt_arch_spec_t* spec_by_architecture(const char* arch) {
    if (!arch || !arch[0]) return NULL;
    if (strstr(arch, "Llama"))    return spec_by_name("llama");
    if (strstr(arch, "Qwen2"))    return spec_by_name("qwen2");
    if (strstr(arch, "Qwen3"))    return spec_by_name("qwen3");
    if (strstr(arch, "Gemma"))    return spec_by_name("gemma");
    if (strstr(arch, "GPT2"))     return spec_by_name("gpt2");
    if (strstr(arch, "Phi3"))     return spec_by_name("phi3");
    if (strstr(arch, "Mixtral"))  return spec_by_name("mixtral");
    if (strstr(arch, "Mistral"))  return spec_by_name("mistral");
    return NULL;
}

const bmt_arch_spec_t* bmt_arch_spec_for_config(const bmt_config_t* cfg) {
    const bmt_arch_spec_t* spec = spec_by_name(cfg->model_type);
    if (!spec) spec = spec_by_architecture(cfg->architecture);
    return spec ? spec : &k_unknown_spec;
}

int bmt_config_to_arch(const bmt_config_t* cfg, const bmt_arch_spec_t* spec,
                       bm_arch_t* arch) {
    if (!spec->supported) return -1;

    if (cfg->hidden_size <= 0 || cfg->intermediate_size <= 0 ||
        cfg->num_hidden_layers <= 0 || cfg->num_attention_heads <= 0 ||
        cfg->vocab_size <= 0) {
        BMT_LOG_ERROR("config missing required dims: hidden=%d ffn=%d layers=%d heads=%d vocab=%d",
                      cfg->hidden_size, cfg->intermediate_size,
                      cfg->num_hidden_layers, cfg->num_attention_heads,
                      cfg->vocab_size);
        return -1;
    }

    memset(arch, 0, sizeof(*arch));
    arch->dim = cfg->hidden_size;
    arch->hidden_dim = cfg->intermediate_size;
    arch->n_layers = cfg->num_hidden_layers;
    arch->n_heads = cfg->num_attention_heads;
    arch->n_kv_heads = cfg->num_key_value_heads > 0
                       ? cfg->num_key_value_heads : cfg->num_attention_heads;
    arch->vocab_size = cfg->vocab_size;
    arch->padded_vocab_size = cfg->vocab_size;
    arch->max_seq_len = cfg->max_position_embeddings > 0
                        ? cfg->max_position_embeddings : 2048;
    arch->head_dim = cfg->head_dim;

    arch->norm = cfg->has_rms_norm_eps ? BM_NORM_RMSNORM : BM_NORM_LAYERNORM;
    arch->activation = (strstr(cfg->hidden_act, "silu") || strstr(cfg->hidden_act, "swiglu"))
                       ? BM_ACT_SWIGLU : BM_ACT_GELU;
    arch->pos_enc = cfg->has_rope_theta ? BM_POS_ROPE : BM_POS_LEARNED;
    arch->attention = arch->n_kv_heads < arch->n_heads ? BM_ATTN_GQA : BM_ATTN_MHA;
    arch->bias = cfg->attention_bias || cfg->mlp_bias;
    arch->weight_tie = cfg->tie_word_embeddings;
    arch->gated_mlp = arch->activation == BM_ACT_SWIGLU;
    arch->rope_theta = cfg->has_rope_theta ? cfg->rope_theta : 10000.0f;
    arch->precision = BM_PRECISION_FP32;
    return 0;
}
