/* config_test.c — typed config parsing + arch family/spec registry. */
#include "baremetal/config.h"
#include "baremetal/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static char* make_config_dir(const char* json) {
    char tmpl[] = "/tmp/bm_cfg_XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) return NULL;

    char* out = malloc(strlen(dir) + 1);
    if (!out) return NULL;
    strcpy(out, dir);

    char path[600];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { free(out); return NULL; }
    fwrite(json, 1, strlen(json), f);
    fclose(f);
    return out;
}

static void free_config_dir(char* dir) {
    if (!dir) return;
    char path[600];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    unlink(path);
    rmdir(dir);
    free(dir);
}

static const char* k_smollm =
    "{"
    "  \"architectures\": [\"LlamaForCausalLM\"],"
    "  \"model_type\": \"llama\","
    "  \"hidden_size\": 576,"
    "  \"intermediate_size\": 1536,"
    "  \"num_hidden_layers\": 30,"
    "  \"num_attention_heads\": 9,"
    "  \"num_key_value_heads\": 3,"
    "  \"vocab_size\": 49152,"
    "  \"max_position_embeddings\": 8192,"
    "  \"rms_norm_eps\": 0.00001,"
    "  \"hidden_act\": \"silu\","
    "  \"rope_theta\": 100000.0,"
    "  \"rope_scaling\": null,"
    "  \"tie_word_embeddings\": true,"
    "  \"attention_bias\": false,"
    "  \"unknown_future_key\": {\"a\": [1, 2, 3]}"
    "}";

static void test_smollm(void) {
    char* dir = make_config_dir(k_smollm);
    check(dir != NULL, "create smollm config dir");
    if (!dir) return;

    bmt_config_t cfg;
    int ok = bmt_config_load(dir, &cfg) == 0;
    check(ok, "load smollm-like config");
    if (ok) {
        check(strcmp(cfg.model_type, "llama") == 0, "model_type");
        check(strcmp(cfg.architecture, "LlamaForCausalLM") == 0, "architecture");
        check(cfg.hidden_size == 576 && cfg.intermediate_size == 1536, "dims");
        check(cfg.num_hidden_layers == 30 && cfg.num_attention_heads == 9, "layers/heads");
        check(cfg.num_key_value_heads == 3 && cfg.vocab_size == 49152, "kv/vocab");
        check(cfg.max_position_embeddings == 8192, "max seq len");
        check(cfg.has_rms_norm_eps == 1, "rms_norm_eps present");
        check(cfg.has_rope_theta == 1 && cfg.rope_theta == 100000.0f, "rope_theta");
        check(cfg.has_rope_scaling == 0, "null rope_scaling ignored");
        check(cfg.tie_word_embeddings == 1, "tie embeddings");
        check(cfg.attention_bias == 0, "attention bias absent");
        check(strcmp(cfg.hidden_act, "silu") == 0, "hidden_act");

        const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
        check(spec->family == BMT_ARCH_FAMILY_LLAMA, "family llama");
        check(spec->supported == 1, "llama supported");

        bm_arch_t arch;
        check(bmt_config_to_arch(&cfg, spec, &arch) == 0, "config -> arch");
        check(arch.dim == 576 && arch.hidden_dim == 1536, "arch dims");
        check(arch.n_layers == 30 && arch.n_heads == 9 && arch.n_kv_heads == 3, "arch layers");
        check(arch.norm == BM_NORM_RMSNORM, "arch norm");
        check(arch.activation == BM_ACT_SWIGLU && arch.gated_mlp == 1, "arch swiglu");
        check(arch.pos_enc == BM_POS_ROPE, "arch rope");
        check(arch.attention == BM_ATTN_GQA, "arch gqa");
        check(arch.weight_tie == 1, "arch tied");
        check(arch.rope_theta == 100000.0f, "arch rope theta");
        check(bmt_arch_validate(&arch, cfg.has_rope_scaling) == 0, "arch validates");
    }

    free_config_dir(dir);
}

static void test_gpt2(void) {
    const char* json =
        "{"
        "  \"model_type\": \"gpt2\","
        "  \"n_embd\": 768,"
        "  \"n_layer\": 12,"
        "  \"n_head\": 12,"
        "  \"n_positions\": 1024,"
        "  \"vocab_size\": 50257,"
        "  \"activation_function\": \"gelu_new\","
        "  \"tie_word_embeddings\": true"
        "}";
    char* dir = make_config_dir(json);
    if (!dir) { check(0, "create gpt2 config"); return; }

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load gpt2 config");
    check(strcmp(cfg.model_type, "gpt2") == 0, "gpt2 model_type");
    check(cfg.hidden_size == 768 && cfg.num_hidden_layers == 12, "gpt2 aliases");
    check(cfg.num_attention_heads == 12 && cfg.max_position_embeddings == 1024, "gpt2 heads/pos");

    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    check(spec->family == BMT_ARCH_FAMILY_GPT2, "gpt2 family");
    check(spec->supported == 0, "gpt2 unsupported");
    bm_arch_t arch;
    check(bmt_config_to_arch(&cfg, spec, &arch) != 0, "gpt2 conversion rejected");

    free_config_dir(dir);
}

static void test_qwen2(void) {
    const char* json =
        "{"
        "  \"model_type\": \"qwen2\","
        "  \"hidden_size\": 896,"
        "  \"intermediate_size\": 4864,"
        "  \"num_hidden_layers\": 24,"
        "  \"num_attention_heads\": 14,"
        "  \"num_key_value_heads\": 2,"
        "  \"vocab_size\": 151936,"
        "  \"max_position_embeddings\": 32768,"
        "  \"rms_norm_eps\": 0.000001,"
        "  \"hidden_act\": \"silu\","
        "  \"rope_theta\": 1000000.0,"
        "  \"attention_bias\": true"
        "}";
    char* dir = make_config_dir(json);
    if (!dir) { check(0, "create qwen2 config"); return; }

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load qwen2 config");
    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    check(spec->family == BMT_ARCH_FAMILY_QWEN2, "qwen2 family");
    check(spec->supported == 0 && spec->unsupported_reason != NULL, "qwen2 unsupported");
    check(strstr(spec->unsupported_reason, "bias") != NULL, "qwen2 reason mentions bias");

    free_config_dir(dir);
}

static void test_gemma_nested(void) {
    const char* json =
        "{"
        "  \"model_type\": \"gemma3\","
        "  \"text_config\": {"
        "    \"hidden_size\": 640,"
        "    \"intermediate_size\": 2048,"
        "    \"num_hidden_layers\": 18,"
        "    \"num_attention_heads\": 8,"
        "    \"num_key_value_heads\": 1,"
        "    \"vocab_size\": 262144,"
        "    \"max_position_embeddings\": 32768,"
        "    \"rms_norm_eps\": 0.000001,"
        "    \"hidden_act\": \"gelu_pytorch_tanh\","
        "    \"rope_theta\": 10000.0"
        "  }"
        "}";
    char* dir = make_config_dir(json);
    if (!dir) { check(0, "create gemma config"); return; }

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load nested gemma config");
    check(strcmp(cfg.model_type, "gemma3") == 0, "gemma root model_type");
    check(cfg.hidden_size == 640 && cfg.num_hidden_layers == 18, "gemma text_config fields");
    check(cfg.num_key_value_heads == 1 && cfg.vocab_size == 262144, "gemma kv/vocab");

    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    check(spec->family == BMT_ARCH_FAMILY_GEMMA, "gemma family");
    check(spec->supported == 0, "gemma unsupported");
    check((spec->weight_transforms & BMT_WEIGHT_TRANSFORM_GEMMA_NORM) != 0,
          "gemma transform hook set");

    free_config_dir(dir);
}

static void test_arch_fallback(void) {
    const char* json =
        "{"
        "  \"architectures\": [\"LlamaForCausalLM\"],"
        "  \"hidden_size\": 64,"
        "  \"intermediate_size\": 128,"
        "  \"num_hidden_layers\": 2,"
        "  \"num_attention_heads\": 4,"
        "  \"num_key_value_heads\": 2,"
        "  \"vocab_size\": 100,"
        "  \"rms_norm_eps\": 0.00001,"
        "  \"hidden_act\": \"silu\","
        "  \"rope_theta\": 10000.0"
        "}";
    char* dir = make_config_dir(json);
    if (!dir) { check(0, "create fallback config"); return; }

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load config without model_type");
    check(cfg.model_type[0] == '\0', "model_type absent");
    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    check(spec->family == BMT_ARCH_FAMILY_LLAMA, "family from architectures");
    check(spec->supported == 1, "fallback family supported");

    free_config_dir(dir);
}

static void test_rope_scaling(void) {
    char json[2048];
    snprintf(json, sizeof(json),
             "{"
             "  \"model_type\": \"llama\","
             "  \"hidden_size\": 64,"
             "  \"intermediate_size\": 128,"
             "  \"num_hidden_layers\": 2,"
             "  \"num_attention_heads\": 4,"
             "  \"num_key_value_heads\": 2,"
             "  \"vocab_size\": 100,"
             "  \"rms_norm_eps\": 0.00001,"
             "  \"hidden_act\": \"silu\","
             "  \"rope_theta\": 500000.0,"
             "  \"rope_scaling\": {\"rope_type\": \"llama3\", \"factor\": 32.0}"
             "}");
    char* dir = make_config_dir(json);
    if (!dir) { check(0, "create rope_scaling config"); return; }

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load active rope_scaling");
    check(cfg.has_rope_scaling == 1, "active rope_scaling detected");

    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    bm_arch_t arch;
    check(bmt_config_to_arch(&cfg, spec, &arch) == 0, "arch conversion");
    check(bmt_arch_validate(&arch, cfg.has_rope_scaling) != 0,
          "rope_scaling rejected by validation");

    free_config_dir(dir);
}

static void test_malformed(void) {
    char* dir = make_config_dir("{\"model_type\": \"llama\",");
    if (!dir) { check(0, "create malformed config"); return; }
    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) != 0, "malformed config rejected");
    free_config_dir(dir);
}

static void test_missing_dims(void) {
    char* dir = make_config_dir("{\"model_type\": \"llama\"}");
    if (!dir) { check(0, "create empty llama config"); return; }
    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load config without dims");
    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    bm_arch_t arch;
    check(bmt_config_to_arch(&cfg, spec, &arch) != 0, "missing dims rejected");
    free_config_dir(dir);
}

static void test_real_config(void) {
    const char* dir = "models/smollm2-135m";
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("SKIP: %s not present\n", path);
        return;
    }
    fclose(f);

    bmt_config_t cfg;
    check(bmt_config_load(dir, &cfg) == 0, "load real SmolLM2 config");
    check(cfg.hidden_size == 576 && cfg.num_hidden_layers == 30, "real config dims");
    check(cfg.vocab_size == 49152 && cfg.num_key_value_heads == 3, "real config vocab/kv");
    check(cfg.has_rope_scaling == 0, "real config null rope_scaling");
    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    check(spec->supported == 1, "real config supported");
}

int main(void) {
    test_smollm();
    test_gpt2();
    test_qwen2();
    test_gemma_nested();
    test_arch_fallback();
    test_rope_scaling();
    test_malformed();
    test_missing_dims();
    test_real_config();

    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all config tests passed\n");
    return 0;
}
