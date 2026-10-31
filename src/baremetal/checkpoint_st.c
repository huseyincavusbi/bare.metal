// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/checkpoint.h"
#include "baremetal/config.h"
#include "baremetal/json.h"
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
    void*           map_base;   // mmap base (for munmap)
    size_t          map_size;
    void*           data;       // data section start
    size_t          data_size;
    int             fd;
    char            path[1024];
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
    if (header_len == 0 || header_len > (1u << 24)) {
        close(fd);
        return NULL;
    }

    char* header = malloc(header_len + 1);
    if (!header) { close(fd); return NULL; }
    if (read(fd, header, header_len) != (ssize_t)header_len) {
        free(header); close(fd); return NULL;
    }
    header[header_len] = '\0';

    struct stat st;
    if (fstat(fd, &st) != 0) {
        free(header); close(fd); return NULL;
    }
    off_t data_start = (off_t)(8 + header_len);
    if (st.st_size < data_start) {
        free(header); close(fd); return NULL;
    }

    void* base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        free(header); close(fd); return NULL;
    }

    bmt_st_file_t* sf = calloc(1, sizeof(bmt_st_file_t));
    if (!sf) {
        munmap(base, (size_t)st.st_size);
        free(header); close(fd); return NULL;
    }
    bmt_st_parse_header(header, header_len, &sf->entries, &sf->n_entries);
    free(header);
    if (sf->n_entries <= 0) {
        munmap(base, (size_t)st.st_size);
        free(sf->entries);
        free(sf); close(fd);
        return NULL;
    }

    sf->map_base = base;
    sf->map_size = (size_t)st.st_size;
    sf->data = (char*)base + data_start;
    sf->data_size = (size_t)(st.st_size - data_start);
    sf->fd = fd;
    snprintf(sf->path, sizeof(sf->path), "%s", path);
    return sf;
}

static void bmt_st_close(bmt_st_file_t* sf) {
    if (!sf) return;
    if (sf->map_base && sf->map_base != MAP_FAILED) {
        munmap(sf->map_base, sf->map_size);
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

typedef struct {
    bmt_st_file_t** files;
    int count;
    int cap;
} bmt_st_set_t;

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

static char* dup_name(const char* s) {
    size_t n = strlen(s) + 1;
    char* p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int cmp_name(const void* a, const void* b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

static int bmt_st_set_add(bmt_st_set_t* set, const char* path) {
    if (set->count == set->cap) {
        int cap = set->cap ? set->cap * 2 : 8;
        bmt_st_file_t** files = realloc(set->files, (size_t)cap * sizeof(*files));
        if (!files) return -1;
        set->files = files;
        set->cap = cap;
    }
    bmt_st_file_t* sf = bmt_st_open(path);
    if (!sf) return -1;
    set->files[set->count++] = sf;
    return 0;
}

static void bmt_st_set_close(bmt_st_set_t* set) {
    for (int i = 0; i < set->count; i++) bmt_st_close(set->files[i]);
    free(set->files);
    set->files = NULL;
    set->count = 0;
    set->cap = 0;
}

static int bmt_st_set_open(bmt_st_set_t* set, const char* dir_path) {
    memset(set, 0, sizeof(*set));

    char index_path[1024];
    snprintf(index_path, sizeof(index_path), "%s/model.safetensors.index.json", dir_path);
    char* text = read_text_file(index_path);
    if (text) {
        char err[160];
        bmt_json_t* root = bmt_json_parse(text, err, sizeof(err));
        free(text);
        if (!root) {
            BMT_LOG_ERROR("Invalid index JSON %s: %s", index_path, err);
            return -1;
        }
        const bmt_json_t* map = bmt_json_get(root, "weight_map");
        if (bmt_json_type(map) != BMT_JSON_OBJECT) {
            BMT_LOG_ERROR("%s: missing weight_map", index_path);
            bmt_json_free(root);
            return -1;
        }
        int n = bmt_json_object_count(map);
        for (int i = 0; i < n; i++) {
            const char* shard = bmt_json_string(bmt_json_object_value(map, i));
            if (!shard) continue;
            char path[2048];
            snprintf(path, sizeof(path), "%s/%s", dir_path, shard);
            int seen = 0;
            for (int j = 0; j < set->count; j++) {
                if (strcmp(set->files[j]->path, path) == 0) { seen = 1; break; }
            }
            if (seen) continue;
            if (bmt_st_set_add(set, path) != 0) {
                BMT_LOG_ERROR("Cannot open shard: %s", path);
                bmt_json_free(root);
                bmt_st_set_close(set);
                return -1;
            }
        }
        bmt_json_free(root);
        if (set->count == 0) {
            BMT_LOG_ERROR("%s: weight_map references no shards", index_path);
            return -1;
        }
        return 0;
    }

    DIR* d = opendir(dir_path);
    if (!d) {
        BMT_LOG_ERROR("Cannot open directory: %s", dir_path);
        return -1;
    }

    char** names = NULL;
    int n_names = 0, cap = 0;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        size_t len = strlen(de->d_name);
        if (len < 12 || strcmp(de->d_name + len - 12, ".safetensors") != 0)
            continue;
        if (n_names == cap) {
            int next = cap ? cap * 2 : 8;
            char** grown = realloc(names, (size_t)next * sizeof(*grown));
            if (!grown) break;
            names = grown;
            cap = next;
        }
        names[n_names] = dup_name(de->d_name);
        if (!names[n_names]) break;
        n_names++;
    }
    closedir(d);

    if (n_names == 0) {
        BMT_LOG_ERROR("No safetensors files found in %s", dir_path);
        free(names);
        return -1;
    }

    qsort(names, (size_t)n_names, sizeof(*names), cmp_name);
    int rc = 0;
    for (int i = 0; i < n_names; i++) {
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", dir_path, names[i]);
        if (bmt_st_set_add(set, path) != 0) {
            BMT_LOG_ERROR("Cannot open safetensors file: %s", path);
            rc = -1;
            break;
        }
    }
    for (int i = 0; i < n_names; i++) free(names[i]);
    free(names);
    if (rc != 0) {
        bmt_st_set_close(set);
        return -1;
    }
    return 0;
}

static void* bmt_st_set_find(const bmt_st_set_t* set, const char* name,
                             size_t* out_size, int* out_is_bf16) {
    for (int i = 0; i < set->count; i++) {
        void* p = bmt_st_get_tensor(set->files[i], name, out_size, out_is_bf16);
        if (p) return p;
    }
    return NULL;
}

static int bmt_st_set_has(const bmt_st_set_t* set, const char* name) {
    size_t sz;
    int bf16;
    return bmt_st_set_find(set, name, &sz, &bf16) != NULL;
}

static float bf16_to_f32(uint16_t bf16);

typedef struct {
    char names[64][192];
    int count;
    int overflow;
} bmt_missing_t;

static void missing_add(bmt_missing_t* m, const char* name) {
    int cap = (int)(sizeof(m->names) / sizeof(m->names[0]));
    if (m->count < cap) {
        snprintf(m->names[m->count], sizeof(m->names[0]), "%s", name);
        m->count++;
    } else {
        m->overflow = 1;
    }
}

static int bmt_st_load_tensor(const bmt_st_set_t* set, const char* name,
                              float* dst, size_t count, bmt_missing_t* missing,
                              int* errors, int* loaded) {
    size_t sz = 0;
    int bf16 = 0;
    void* raw = bmt_st_set_find(set, name, &sz, &bf16);
    if (!raw) {
        missing_add(missing, name);
        return -1;
    }

    size_t expected = bf16 ? count * 2 : count * 4;
    if (sz != expected) {
        BMT_LOG_ERROR("tensor %s: %zu bytes, expected %zu (%s)",
                      name, sz, expected, bf16 ? "bf16" : "f32");
        (*errors)++;
        return -1;
    }

    if (bf16) {
        const uint16_t* src = (const uint16_t*)raw;
        for (size_t i = 0; i < count; i++) dst[i] = bf16_to_f32(src[i]);
    } else {
        memcpy(dst, raw, count * sizeof(float));
    }
    (*loaded)++;
    return 0;
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

    // Load safetensors files (single file, or shards via index.json)
    bmt_st_set_t set;
    if (bmt_st_set_open(&set, dir_path) != 0) return -1;

    // Detect qk_norm, ffn_post_norm, gated_mlp, weight_tie from tensor names
    arch.has_qk_norm = 0;
    arch.has_ffn_post_norm = 0;
    int saw_lm_head = 0;
    for (int i = 0; i < set.count; i++) {
        for (int j = 0; j < set.files[i]->n_entries; j++) {
            const char* n = set.files[i]->entries[j].name;
            if (strstr(n, ".q_norm")) arch.has_qk_norm = 1;
            if (strstr(n, ".up_proj")) arch.gated_mlp = 1;
            if (strstr(n, "post_feedforward")) arch.has_ffn_post_norm = 1;
            if (strstr(n, "lm_head")) saw_lm_head = 1;
        }
    }
    if (!saw_lm_head) arch.weight_tie = 1;

    if (bmt_arch_validate(&arch, cfg.has_rope_scaling) != 0) {
        bmt_st_set_close(&set);
        return -1;
    }

    // Allocate model buffers
    if (bmt_model_alloc_buffers(model, &arch) != 0) {
        bmt_st_set_close(&set);
        return -1;
    }

    int L = arch.n_layers, D = arch.dim, H = arch.hidden_dim, V = arch.vocab_size;
    int NH = arch.n_heads, HD = model->head_size;
    float* w = (float*)model->weight_buffer;

    bmt_missing_t missing;
    memset(&missing, 0, sizeof(missing));
    int load_errors = 0;
    int loaded = 0;

    #define CPT(name, ptr, count) do { \
        (void)bmt_st_load_tensor(&set, (name), (ptr), (size_t)(count), \
                                 &missing, &load_errors, &loaded); \
    } while (0)

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
        if (bmt_st_set_has(&set, t->lm_head)) {
            CPT(t->lm_head, w, V * D);
        } else {
            BMT_LOG_WARN("lm_head missing; using embedding weights");
            CPT(t->embed, w, V * D);
        }
        w += V * D;
    }

    int shards = set.count;
    int missing_total = missing.count + (missing.overflow ? 1 : 0);

    if (missing_total > 0 || load_errors > 0) {
        BMT_LOG_ERROR("Failed to load %s: %d missing, %d size errors",
                      dir_path, missing_total, load_errors);
        for (int i = 0; i < missing.count; i++)
            BMT_LOG_ERROR("  missing: %s", missing.names[i]);
        if (missing.overflow)
            BMT_LOG_ERROR("  ... more missing tensors not listed");
        bmt_st_set_close(&set);
        bmt_model_free_buffers(model);
        return -1;
    }

    bmt_st_set_close(&set);
    BMT_LOG_INFO("Loaded safetensors: %s (%zu params, %d shard(s), %d tensors)",
                 dir_path, model->n_parameters, shards, loaded);
    return 0;
}
