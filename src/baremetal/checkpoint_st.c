// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/checkpoint.h"
#include "baremetal/config.h"
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

static void tensor_name(char* out, size_t out_len, const char* prefix,
                        int layer, const char* suffix) {
    snprintf(out, out_len, "%s%d.%s", prefix, layer, suffix);
}

int bmt_checkpoint_load_safetensors(bm_model_t* model, const char* dir_path) {
    bmt_config_t cfg;
    if (bmt_config_load(dir_path, &cfg) != 0) return -1;

    const bmt_arch_spec_t* spec = bmt_arch_spec_for_config(&cfg);
    if (!spec->supported) {
        BMT_LOG_ERROR("Unsupported architecture '%s': %s",
                      cfg.model_type[0] ? cfg.model_type : "(missing model_type)",
                      spec->unsupported_reason ? spec->unsupported_reason : "not implemented");
        return -1;
    }

    bm_arch_t arch;
    if (bmt_config_to_arch(&cfg, spec, &arch) != 0) return -1;

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

    if (bmt_arch_validate(&arch, cfg.has_rope_scaling) != 0) {
        for (int i = 0; i < n_st; i++) bmt_st_close(st_files[i]);
        return -1;
    }

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

    const bmt_tensor_map_t* t = spec->tensors;

    CPT(t->embed, w, V * D);
    w += V * D;

    // Group weights by type to match model.c layout
    char buf[256];

    // ln1w: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->input_norm);
        CPT(buf, w, D); w += D;
    }

    // qw: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->q);
        CPT(buf, w, NH * HD * D); w += NH * HD * D;
    }

    // kw: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->k);
        CPT(buf, w, model->n_kv_heads * HD * D); w += model->n_kv_heads * HD * D;
    }

    // vw: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->v);
        CPT(buf, w, model->n_kv_heads * HD * D); w += model->n_kv_heads * HD * D;
    }

    // QK norm
    if (arch.has_qk_norm) {
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->q_norm);
            CPT(buf, w, HD); w += HD;
        }
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->k_norm);
            CPT(buf, w, model->n_kv_heads * HD); w += model->n_kv_heads * HD;
        }
    }

    // attprojw: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->o);
        CPT(buf, w, D * NH * HD); w += NH * HD * D;
    }

    // ln2w: all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->post_attn_norm);
        CPT(buf, w, D); w += D;
    }

    if (arch.gated_mlp) {
        // fcw (gate): all layers
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->gate);
            CPT(buf, w, H * D); w += H * D;
        }

        // fcw3 (up): all layers
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->up);
            CPT(buf, w, H * D); w += H * D;
        }
    }

    // fcprojw (down): all layers
    for (int l = 0; l < L; l++) {
        tensor_name(buf, sizeof(buf), t->layers, l, t->down);
        CPT(buf, w, D * H); w += D * H;
    }

    if (arch.has_ffn_post_norm) {
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->pre_ffn_norm);
            CPT(buf, w, D); w += D;
        }
        for (int l = 0; l < L; l++) {
            tensor_name(buf, sizeof(buf), t->layers, l, t->post_ffn_norm);
            CPT(buf, w, D); w += D;
        }
    }

    // Final norm
    CPT(t->final_norm, w, D); w += D;

    if (!arch.weight_tie) {
        CPT(t->lm_head, w, V * D);
        CPT(t->embed, w, V * D); // fallback
        w += V * D;
    }

    for (int i = 0; i < n_st; i++) bmt_st_close(st_files[i]);

    BMT_LOG_INFO("Loaded safetensors: %s (%zu params)", dir_path, model->n_parameters);
    return 0;
}
