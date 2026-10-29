/* arch_validate_test.c — unsupported architecture features must fail fast
 * instead of loading and silently producing wrong logits. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bm_arch_t base_arch(void) {
    bm_arch_t a;
    memset(&a, 0, sizeof(a));
    a.dim = 64;
    a.hidden_dim = 128;
    a.n_layers = 2;
    a.n_heads = 4;
    a.n_kv_heads = 2;
    a.vocab_size = 128;
    a.padded_vocab_size = 128;
    a.max_seq_len = 128;
    a.norm = BM_NORM_RMSNORM;
    a.activation = BM_ACT_SWIGLU;
    a.pos_enc = BM_POS_ROPE;
    a.attention = BM_ATTN_GQA;
    a.precision = BM_PRECISION_FP32;
    a.head_dim = 16;
    a.gated_mlp = 1;
    a.weight_tie = 1;
    a.rope_theta = 10000.0f;
    return a;
}

static int expect_reject(const char* name, bm_arch_t arch, int has_rope_scaling) {
    int rc = bmt_arch_validate(&arch, has_rope_scaling);
    printf("%s: %s rejected\n", rc != 0 ? "PASS" : "FAIL", name);
    return rc != 0 ? 0 : 1;
}

static int bin_checkpoint_rejects_bias(void) {
    char path[] = "/tmp/bm_arch_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        printf("FAIL: mkstemp\n");
        return 1;
    }

    int32_t h[256];
    memset(h, 0, sizeof(h));
    h[0]  = BMT_CHECKPOINT_MAGIC;
    h[2]  = BM_PRECISION_FP32;
    h[3]  = 64;
    h[4]  = 128;
    h[5]  = 2;
    h[6]  = 4;
    h[7]  = 2;
    h[8]  = 128;
    h[9]  = 128;
    h[10] = BM_NORM_RMSNORM;
    h[11] = BM_ACT_SWIGLU;
    h[12] = BM_POS_ROPE;
    h[13] = BM_ATTN_GQA;
    h[14] = 1;
    h[15] = 1;
    h[18] = 16;
    h[19] = 1;
    h[23] = 10000;

    FILE* f = fdopen(fd, "wb");
    if (!f) {
        printf("FAIL: fdopen\n");
        close(fd);
        unlink(path);
        return 1;
    }
    fwrite(h, sizeof(h), 1, f);
    fclose(f);

    bm_model_t model;
    memset(&model, 0, sizeof(model));
    int rc = bmt_checkpoint_load(&model, path);
    unlink(path);

    printf("%s: bin checkpoint with attention_bias rejected\n",
           rc != 0 ? "PASS" : "FAIL");
    return rc != 0 ? 0 : 1;
}

int main(void) {
    int failures = 0;

    bm_arch_t arch = base_arch();
    int rc = bmt_arch_validate(&arch, 0);
    printf("%s: supported arch accepted\n", rc == 0 ? "PASS" : "FAIL");
    if (rc != 0) failures++;

    arch = base_arch();
    arch.bias = 1;
    failures += expect_reject("attention_bias", arch, 0);

    arch = base_arch();
    arch.has_qk_norm = 1;
    failures += expect_reject("qk_norm", arch, 0);

    arch = base_arch();
    arch.has_ffn_post_norm = 1;
    failures += expect_reject("post_ffn_norm", arch, 0);

    arch = base_arch();
    arch.gemma_norm = 1;
    failures += expect_reject("gemma_norm", arch, 0);

    arch = base_arch();
    arch.embed_scale = 1;
    failures += expect_reject("embed_scale", arch, 0);

    failures += expect_reject("rope_scaling", base_arch(), 1);

    arch = base_arch();
    arch.bias = 1;
    arch.has_qk_norm = 1;
    failures += expect_reject("multiple features", arch, 1);

    failures += bin_checkpoint_rejects_bias();

    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all arch validation tests passed\n");
    return 0;
}
