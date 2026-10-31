/* sharded_load_test.c — safetensors sharding (index.json) and missing tensors. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#define D   32
#define H   64
#define L   2
#define NH  2
#define NKV 1
#define HD  16
#define V   64

typedef struct {
    char name[128];
    int rows;
    int cols;
    int count;
} tdesc_t;

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static void add1(tdesc_t* t, int* n, const char* name, int count) {
    snprintf(t[*n].name, sizeof(t[*n].name), "%s", name);
    t[*n].rows = count;
    t[*n].cols = 0;
    t[*n].count = count;
    (*n)++;
}

static void add2(tdesc_t* t, int* n, const char* name, int rows, int cols) {
    snprintf(t[*n].name, sizeof(t[*n].name), "%s", name);
    t[*n].rows = rows;
    t[*n].cols = cols;
    t[*n].count = rows * cols;
    (*n)++;
}

static int build_tensors(tdesc_t* t) {
    int n = 0;
    add2(t, &n, "model.embed_tokens.weight", V, D);
    for (int l = 0; l < L; l++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
        add1(t, &n, buf, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", l);
        add2(t, &n, buf, NH * HD, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", l);
        add2(t, &n, buf, NKV * HD, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", l);
        add2(t, &n, buf, NKV * HD, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", l);
        add2(t, &n, buf, D, NH * HD);
        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
        add1(t, &n, buf, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", l);
        add2(t, &n, buf, H, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", l);
        add2(t, &n, buf, H, D);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", l);
        add2(t, &n, buf, D, H);
    }
    add1(t, &n, "model.norm.weight", D);
    return n;
}

static float weight_value(const char* name, int i) {
    uint64_t h = 1469598103934665603ull;
    for (const char* p = name; *p; p++) {
        h ^= (unsigned char)*p;
        h *= 1099511628211ull;
    }
    h ^= (uint64_t)i * 2654435761ull;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return (float)((h & 0xFFFF) / 65535.0) * 0.2f - 0.1f;
}

static int write_shard(const char* path, const tdesc_t* all,
                       const int* sel, int nsel) {
    char header[65536];
    size_t off = 0;
    size_t data_off = 0;

    header[off++] = '{';
    for (int i = 0; i < nsel; i++) {
        const tdesc_t* t = &all[sel[i]];
        size_t bytes = (size_t)t->count * sizeof(float);
        if (i) header[off++] = ',';
        if (t->cols > 0) {
            off += (size_t)snprintf(header + off, sizeof(header) - off,
                "\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                t->name, t->rows, t->cols, data_off, data_off + bytes);
        } else {
            off += (size_t)snprintf(header + off, sizeof(header) - off,
                "\"%s\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%zu,%zu]}",
                t->name, t->rows, data_off, data_off + bytes);
        }
        data_off += bytes;
    }
    header[off++] = '}';

    size_t padded = (off + 7u) & ~(size_t)7u;
    while (off < padded) header[off++] = ' ';

    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t hlen = (uint64_t)padded;
    if (fwrite(&hlen, sizeof(hlen), 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(header, 1, padded, f) != padded) { fclose(f); return -1; }

    for (int i = 0; i < nsel; i++) {
        const tdesc_t* t = &all[sel[i]];
        float* buf = malloc((size_t)t->count * sizeof(float));
        if (!buf) { fclose(f); return -1; }
        for (int j = 0; j < t->count; j++)
            buf[j] = weight_value(t->name, j);
        size_t wrote = fwrite(buf, sizeof(float), (size_t)t->count, f);
        free(buf);
        if (wrote != (size_t)t->count) { fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

static void write_config(const char* dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f,
        "{\"model_type\":\"llama\",\"hidden_size\":%d,\"intermediate_size\":%d,"
        "\"num_hidden_layers\":%d,\"num_attention_heads\":%d,\"num_key_value_heads\":%d,"
        "\"vocab_size\":%d,\"max_position_embeddings\":128,\"rms_norm_eps\":0.00001,"
        "\"hidden_act\":\"silu\",\"rope_theta\":10000.0,\"tie_word_embeddings\":true}",
        D, H, L, NH, NKV, V);
    fclose(f);
}

static void write_index(const char* dir, const tdesc_t* t,
                        const int* shard_of, int n, const char* s1, const char* s2) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "{\"metadata\":{},\"weight_map\":{");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s\"%s\":\"%s\"", i ? "," : "", t[i].name,
                shard_of[i] == 0 ? s1 : s2);
    }
    fprintf(f, "}}\n");
    fclose(f);
}


static void cleanup(const char* base) {
    const char* files[] = {
        "single/config.json", "single/model.safetensors",
        "sharded/config.json", "sharded/model-00001-of-00002.safetensors",
        "sharded/model-00002-of-00002.safetensors",
        "sharded/model.safetensors.index.json",
        "missing/config.json", "missing/model-00001-of-00002.safetensors",
        "missing/model-00002-of-00002.safetensors",
        "missing/model.safetensors.index.json",
    };
    char path[1024];
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", base, files[i]);
        unlink(path);
    }
    const char* dirs[] = { "single", "sharded", "missing" };
    for (size_t i = 0; i < sizeof(dirs) / sizeof(dirs[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", base, dirs[i]);
        rmdir(path);
    }
    rmdir(base);
}

int main(void) {
    char base[] = "/tmp/bm_shard_XXXXXX";
    if (!mkdtemp(base)) {
        printf("FAIL: mkdtemp\n");
        return 1;
    }

    char single[1024], sharded[1024], missing[1024];
    snprintf(single, sizeof(single), "%s/single", base);
    snprintf(sharded, sizeof(sharded), "%s/sharded", base);
    snprintf(missing, sizeof(missing), "%s/missing", base);
    mkdir(single, 0700);
    mkdir(sharded, 0700);
    mkdir(missing, 0700);

    tdesc_t tensors[64];
    int n = build_tensors(tensors);

    write_config(single);
    write_config(sharded);
    write_config(missing);

    int all[64];
    for (int i = 0; i < n; i++) all[i] = i;

    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", single);
    check(write_shard(path, tensors, all, n) == 0, "write single-file shard");

    int s1[64], s2[64], n1 = 0, n2 = 0;
    int shard_of[64];
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) { s1[n1++] = i; shard_of[i] = 0; }
        else            { s2[n2++] = i; shard_of[i] = 1; }
    }
    char p1[1024], p2[1024];
    snprintf(p1, sizeof(p1), "%s/model-00001-of-00002.safetensors", sharded);
    snprintf(p2, sizeof(p2), "%s/model-00002-of-00002.safetensors", sharded);
    check(write_shard(p1, tensors, s1, n1) == 0, "write shard 1");
    check(write_shard(p2, tensors, s2, n2) == 0, "write shard 2");
    write_index(sharded, tensors, shard_of, n,
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors");

    int missing_idx = -1;
    for (int i = 0; i < n; i++)
        if (strcmp(tensors[i].name, "model.layers.0.self_attn.q_proj.weight") == 0)
            missing_idx = i;

    int m1[64], m2[64], nm1 = 0, nm2 = 0;
    for (int i = 0; i < n; i++) {
        if (i == missing_idx) continue;
        if (i % 2 == 0) m1[nm1++] = i;
        else            m2[nm2++] = i;
    }
    char q1[1024], q2[1024];
    snprintf(q1, sizeof(q1), "%s/model-00001-of-00002.safetensors", missing);
    snprintf(q2, sizeof(q2), "%s/model-00002-of-00002.safetensors", missing);
    check(write_shard(q1, tensors, m1, nm1) == 0, "write missing shard 1");
    check(write_shard(q2, tensors, m2, nm2) == 0, "write missing shard 2");
    write_index(missing, tensors, shard_of, n,
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors");

    bm_model_t ma, mb;
    memset(&ma, 0, sizeof(ma));
    memset(&mb, 0, sizeof(mb));

    int rc_single = bmt_checkpoint_load_safetensors(&ma, single);
    int rc_shard = bmt_checkpoint_load_safetensors(&mb, sharded);
    check(rc_single == 0, "single-file load");
    check(rc_shard == 0, "two-shard load");
    check(ma.n_parameters == mb.n_parameters, "parameter count matches");
    if (rc_single == 0 && rc_shard == 0 && ma.weight_buffer && mb.weight_buffer) {
        size_t bytes = ma.n_parameters * sizeof(float);
        check(memcmp(ma.weight_buffer, mb.weight_buffer, bytes) == 0,
              "sharded weights bit-identical to single file");
    }

    bm_model_t mc;
    memset(&mc, 0, sizeof(mc));
    int rc_missing = bmt_checkpoint_load_safetensors(&mc, missing);
    check(rc_missing != 0, "missing tensor rejected");
    check(mc.weight_buffer == NULL, "failed load releases buffers");

    bmt_model_free_buffers(&ma);
    bmt_model_free_buffers(&mb);
    bmt_model_free_buffers(&mc);

    cleanup(base);

    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all sharded-load tests passed\n");
    return 0;
}
