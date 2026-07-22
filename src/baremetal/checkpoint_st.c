#include "baremetal/checkpoint.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

// minimal safetensors JSON header parser
// expects: {"tensor_name": {"dtype":"F32","shape":[a,b],"data_offsets":[start,end]}, ...}

typedef struct {
    char   name[256];
    size_t data_offset;
    size_t data_length;
    int    n_dims;
    int    shape[4];
    int    is_bf16;
} bmt_st_entry_t;

typedef struct {
    bmt_st_entry_t* entries;
    int             n_entries;
    void*           data;       // mmap'd data
    size_t          data_size;
    int             fd;
} bmt_st_file_t;

static int bmt_st_parse_header(const char* json, size_t len, bmt_st_entry_t** out_entries, int* out_n) {
    int cap = 128;
    bmt_st_entry_t* entries = malloc(cap * sizeof(bmt_st_entry_t));
    int n = 0;
    const char* p = json;
    const char* end = json + len;

    while (p < end && *p != '{') p++;

    while (p < end) {
        while (p < end && *p != '"' && *p != '}') p++;
        if (p >= end || *p == '}') break;

        p++; // skip opening "
        char* name = entries[n].name;
        while (p < end && *p != '"' && (name - entries[n].name) < 255) *name++ = *p++;
        *name = '\0';
        p++; // skip closing "

        if (strncmp(entries[n].name, "__", 2) == 0) {
            const char* skip = strstr(p, "}");
            if (!skip) break;
            p = skip + 1;
            continue;
        }

        entries[n].is_bf16 = 0;
        const char* dt = strstr(p, "\"dtype\"");
        if (dt) {
            dt = strstr(dt, ":");
            if (dt && strstr(dt, "BF16")) entries[n].is_bf16 = 1;
        }

        // find data_offsets
        const char* off = strstr(p, "data_offsets");
        if (!off) break;
        off = strstr(off, "[");
        if (!off) break;
        size_t start, end_off;
        if (sscanf(off, "[%zu,%zu]", &start, &end_off) != 2) break;

        entries[n].data_offset = start;
        entries[n].data_length = end_off - start;

        // find shape
        const char* sh = strstr(p, "shape");
        entries[n].n_dims = 0;
        if (sh) {
            sh = strstr(sh, "[");
            if (sh) {
                sh++;
                while (*sh && *sh != ']' && entries[n].n_dims < 4) {
                    while (*sh == ' ' || *sh == ',') sh++;
                    if (*sh == ']') break;
                    entries[n].shape[entries[n].n_dims++] = (int)strtol(sh, (char**)&sh, 10);
                }
            }
        }

        p = off;
        while (p < end && *p != ']') p++;
        if (p < end && *p == ']') p++;
        if (p < end && *p == '}') p++;
        n++;
        if (n >= cap) {
            cap *= 2;
            entries = realloc(entries, cap * sizeof(bmt_st_entry_t));
        }
        if (p < end && *p == ',') p++;
        if (p >= end || *p == '}') break;
    }

    *out_entries = entries;
    *out_n = n;
    return 0;
}

static bmt_st_file_t* bmt_st_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    uint64_t header_len;
    if (read(fd, &header_len, 8) != 8) { close(fd); return NULL; }

    char* header = malloc(header_len + 1);
    if (read(fd, header, header_len) != (ssize_t)header_len) {
        free(header); close(fd); return NULL;
    }
    header[header_len] = '\0';

    bmt_st_file_t* sf = calloc(1, sizeof(bmt_st_file_t));
    bmt_st_parse_header(header, header_len, &sf->entries, &sf->n_entries);
    free(header);

    off_t data_start = 8 + header_len;
    struct stat st;
    fstat(fd, &st);
    sf->data_size = st.st_size - data_start;
    sf->data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    sf->fd = fd;

    if (sf->data != MAP_FAILED) {
        sf->data = (char*)sf->data + data_start;
    }

    return sf;
}

static void bmt_st_close(bmt_st_file_t* sf) {
    if (!sf) return;
    if (sf->data && sf->data != MAP_FAILED) {
        munmap(sf->data, sf->data_size);
    }
    close(sf->fd);
    free(sf->entries);
    free(sf);
}

static void* bmt_st_get_tensor(bmt_st_file_t* sf, const char* name, size_t* out_size, int* out_is_bf16) {
    for (int i = 0; i < sf->n_entries; i++) {
        if (strcmp(sf->entries[i].name, name) == 0) {
            *out_size = sf->entries[i].data_length;
            if (out_is_bf16) *out_is_bf16 = sf->entries[i].is_bf16;
            return (char*)sf->data + sf->entries[i].data_offset;
        }
    }
    return NULL;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

int bmt_checkpoint_load_safetensors(bm_model_t* model, const char* dir_path) {
    // Read config.json
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%s/config.json", dir_path);
    FILE* cf = fopen(config_path, "r");
    if (!cf) { BMT_LOG_ERROR("Cannot open %s", config_path); return -1; }
    fseek(cf, 0, SEEK_END);
    long cflen = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    char* config_json = malloc(cflen + 1);
    fread(config_json, 1, cflen, cf);
    fclose(cf);
    config_json[cflen] = '\0';

    // Minimal JSON parsing for key config values
    bm_arch_t arch;
    memset(&arch, 0, sizeof(arch));

    #define JINT(key, field) do { \
        char* p = strstr(config_json, "\"" key "\""); \
        if (p) { p = strstr(p, ":"); if (p) arch.field = (int)strtol(p+1, NULL, 10); } \
    } while(0)

    JINT("hidden_size", dim);
    JINT("intermediate_size", hidden_dim);
    JINT("num_hidden_layers", n_layers);
    JINT("num_attention_heads", n_heads);
    JINT("num_key_value_heads", n_kv_heads);
    if (arch.n_kv_heads == 0) arch.n_kv_heads = arch.n_heads;
    JINT("vocab_size", vocab_size);
    JINT("max_position_embeddings", max_seq_len);
    if (arch.max_seq_len == 0) JINT("n_positions", max_seq_len);
    if (arch.max_seq_len == 0) arch.max_seq_len = 2048;

    arch.padded_vocab_size = arch.vocab_size;

    // RMSNorm detection
    arch.norm = (strstr(config_json, "rms_norm_eps") != NULL) ? BM_NORM_RMSNORM : BM_NORM_LAYERNORM;

    // Activation detection
    char* act = strstr(config_json, "hidden_activation");
    if (!act) act = strstr(config_json, "hidden_act");
    if (act) {
        act = strstr(act, ":");
        if (act) {
            if (strstr(act, "silu") || strstr(act, "swiglu"))
                arch.activation = BM_ACT_SWIGLU;
            else
                arch.activation = BM_ACT_GELU;
        }
    }

    // Position encoding
    arch.pos_enc = (strstr(config_json, "rope_theta") != NULL) ? BM_POS_ROPE : BM_POS_LEARNED;

    // Attention type
    arch.attention = (arch.n_kv_heads < arch.n_heads) ? BM_ATTN_GQA : BM_ATTN_MHA;

    // Bias
    arch.bias = (strstr(config_json, "\"attention_bias\": true") != NULL) ? 1 : 0;

    // Head dim detection
    arch.head_dim = 0;
    char* hd = strstr(config_json, "\"head_dim\"");
    if (hd) { hd = strstr(hd, ":"); if (hd) arch.head_dim = (int)strtol(hd+1, NULL, 10); }

    // Weight tie - prefer config, fall back to safetensors presence
    arch.weight_tie = (strstr(config_json, "\"tie_word_embeddings\": true") != NULL) ? 1 : 0;
    arch.gemma_norm = (strstr(config_json, "\"gemma") != NULL) ? 1 : 0;

    char* rt = strstr(config_json, "\"rope_theta\"");
    if (rt) { rt = strstr(rt, ":"); if (rt) arch.rope_theta = strtof(rt+1, NULL); }
    if (arch.rope_theta <= 0.0f) arch.rope_theta = 10000.0f;
    arch.precision = BM_PRECISION_FP32;

    free(config_json);

    // Load safetensors files
    bmt_st_file_t* st_files[16];
    int n_st = 0;
    DIR* d = opendir(dir_path);
    if (!d) { BMT_LOG_ERROR("Cannot open directory: %s", dir_path); return -1; }
    struct dirent* de;
    while ((de = readdir(d)) != NULL && n_st < 16) {
        if (strstr(de->d_name, ".safetensors")) {
            char fpath[1024];
            snprintf(fpath, sizeof(fpath), "%s/%s", dir_path, de->d_name);
            st_files[n_st] = bmt_st_open(fpath);
            if (st_files[n_st]) n_st++;
        }
    }
    closedir(d);

    if (n_st == 0) { BMT_LOG_ERROR("No safetensors files found"); return -1; }

    // Detect qk_norm, ffn_post_norm, gated_mlp, weight_tie from tensor names
    arch.has_qk_norm = 0;
    arch.has_ffn_post_norm = 0;
    arch.gated_mlp = 0;
    int saw_lm_head = 0;
    for (int i = 0; i < n_st; i++) {
        for (int j = 0; j < st_files[i]->n_entries; j++) {
            const char* n = st_files[i]->entries[j].name;
            if (strstr(n, ".q_norm")) arch.has_qk_norm = 1;
            if (strstr(n, ".up_proj")) arch.gated_mlp = 1;
            if (strstr(n, "post_feedforward")) arch.has_ffn_post_norm = 1;
            if (strstr(n, "lm_head")) saw_lm_head = 1;
        }
    }
    if (!saw_lm_head) arch.weight_tie = 1;

    // Allocate model buffers
    if (bmt_model_alloc_buffers(model, &arch) != 0) {
        for (int i = 0; i < n_st; i++) bmt_st_close(st_files[i]);
        return -1;
    }

    int L = arch.n_layers, D = arch.dim, H = arch.hidden_dim, V = arch.vocab_size;
    int NH = arch.n_heads, HD = model->head_size;
    float* w = (float*)model->weight_buffer;
    size_t sz;

    #define CPT(name, ptr, count) do { \
        for (int _i = 0; _i < n_st; _i++) { \
            int _bf16 = 0; \
            void* _raw = bmt_st_get_tensor(st_files[_i], name, &sz, &_bf16); \
            if (_raw) { \
                if (_bf16) { \
                    uint16_t* _src = (uint16_t*)_raw; \
                    size_t _n = ((size_t)(count) < sz/2 ? (size_t)(count) : sz/2); \
                    for (size_t _j = 0; _j < _n; _j++) (ptr)[_j] = bf16_to_f32(_src[_j]); \
                } else { \
                    memcpy(ptr, _raw, sz < (count)*sizeof(float) ? sz : (count)*sizeof(float)); \
                } \
                break; \
            } \
        } \
    } while(0)

    CPT("model.embed_tokens.weight", w, V * D);
    w += V * D;

    // Group weights by type to match model.c layout
    char buf[256];

    // ln1w: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
        CPT(buf, w, D); w += D;
    }

    // qw: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", l);
        CPT(buf, w, NH * HD * D); w += NH * HD * D;
    }

    // kw: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", l);
        CPT(buf, w, model->n_kv_heads * HD * D); w += model->n_kv_heads * HD * D;
    }

    // vw: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", l);
        CPT(buf, w, model->n_kv_heads * HD * D); w += model->n_kv_heads * HD * D;
    }

    // QK norm
    if (arch.has_qk_norm) {
        for (int l = 0; l < L; l++) {
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_norm.weight", l);
            CPT(buf, w, HD); w += HD;
        }
        for (int l = 0; l < L; l++) {
            snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_norm.weight", l);
            CPT(buf, w, model->n_kv_heads * HD); w += model->n_kv_heads * HD;
        }
    }

    // attprojw: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", l);
        CPT(buf, w, D * NH * HD); w += NH * HD * D;
    }

    // ln2w: all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
        CPT(buf, w, D); w += D;
    }

    // fcw (gate): all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", l);
        CPT(buf, w, H * D); w += H * D;
    }

    // fcw3 (up): all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", l);
        CPT(buf, w, H * D); w += H * D;
    }

    // fcprojw (down): all layers
    for (int l = 0; l < L; l++) {
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", l);
        CPT(buf, w, D * H); w += D * H;
    }

    if (arch.has_ffn_post_norm) {
        for (int l = 0; l < L; l++) {
            snprintf(buf, sizeof(buf), "model.layers.%d.pre_feedforward_layernorm.weight", l);
            CPT(buf, w, D); w += D;
        }
        for (int l = 0; l < L; l++) {
            snprintf(buf, sizeof(buf), "model.layers.%d.post_feedforward_layernorm.weight", l);
            CPT(buf, w, D); w += D;
        }
    }

    // Final norm
    CPT("model.norm.weight", w, D); w += D;

    if (!arch.weight_tie) {
        CPT("lm_head.weight", w, V * D);
        CPT("model.embed_tokens.weight", w, V * D); // fallback
        w += V * D;
    }

    for (int i = 0; i < n_st; i++) bmt_st_close(st_files[i]);

    BMT_LOG_INFO("Loaded safetensors: %s (%zu params)", dir_path, model->n_parameters);
    return 0;
}
