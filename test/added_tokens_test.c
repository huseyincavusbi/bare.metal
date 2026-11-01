/* added_tokens_test.c -- added/special tokens must encode atomically. */
#include "baremetal.h"
#include "baremetal/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static char* make_tokenizer_dir(const char* json) {
    char tmpl[] = "/tmp/bm_tok_XXXXXX";
    char* dir = mkdtemp(tmpl);
    if (!dir) return NULL;
    char* out = malloc(strlen(dir) + 1);
    if (!out) return NULL;
    strcpy(out, dir);
    char path[600];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { free(out); return NULL; }
    fwrite(json, 1, strlen(json), f);
    fclose(f);
    return out;
}

static void free_tokenizer_dir(char* dir) {
    if (!dir) return;
    char path[600];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    unlink(path);
    rmdir(dir);
    free(dir);
}

static void expect_ids(const char* name, bm_tokenizer_t* t, const char* text,
                       const int* want, int nwant) {
    int ids[256];
    int n = 0;
    int rc = bm_tokenizer_encode(t, text, 0, 0, ids, &n);
    int ok = rc == 0 && n == nwant &&
             (nwant == 0 || memcmp(ids, want, (size_t)nwant * sizeof(int)) == 0);
    printf("%s: %s ->", ok ? "PASS" : "FAIL", name);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    if (!ok) {
        printf(" (want");
        for (int i = 0; i < nwant; i++) printf(" %d", want[i]);
        printf(")");
        failures++;
    }
    printf("\n");
}

static void expect_decode(const char* name, bm_tokenizer_t* t,
                          const int* ids, int n, const char* want) {
    char out[512];
    size_t len = 0;
    for (int i = 0; i < n; i++) {
        char* piece = bm_tokenizer_decode(t, i ? ids[i - 1] : 1, ids[i]);
        size_t plen = strlen(piece);
        if (len + plen < sizeof(out)) {
            memcpy(out + len, piece, plen);
            len += plen;
        }
    }
    out[len] = '\0';
    int ok = strcmp(out, want) == 0;
    printf("%s: %s -> \"%s\" (want \"%s\")\n",
           ok ? "PASS" : "FAIL", name, out, want);
    if (!ok) failures++;
}

static void test_synthetic(void) {
    const char* json =
        "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2},"
        "\"merges\":[[\"a\",\"b\"]]},"
        "\"added_tokens\":[{\"id\":3,\"content\":\"<|x|>\",\"special\":true}]}";
    char* dir = make_tokenizer_dir(json);
    if (!dir) { check(0, "synthetic tokenizer dir"); return; }

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, 4);
    int want_ids[] = {0, 3, 1};
    expect_ids("synthetic mixed", &t, "a<|x|>b", want_ids, 3);
    expect_ids("synthetic special", &t, "<|x|>", (int[]){3}, 1);
    expect_decode("synthetic decode", &t, want_ids, 3, "a<|x|>b");
    bm_tokenizer_free(&t);
    free_tokenizer_dir(dir);
}

static void test_real(const char* dir, int vocab_size, const char* name,
                      const char* text, const int* want, int nwant,
                      const char* special, int special_id) {
    if (access(dir, F_OK) != 0) {
        printf("SKIP: %s (missing %s)\n", name, dir);
        return;
    }
    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, vocab_size);
    expect_ids(name, &t, text, want, nwant);
    expect_ids(name, &t, special, &special_id, 1);
    expect_decode(name, &t, want, nwant, text);
    bm_tokenizer_free(&t);
}

int main(void) {
    test_synthetic();

    const char* smol_text = "<|im_start|>user\nhello<|im_end|>";
    int smol_ids[] = {1, 4093, 198, 28120, 2};
    test_real("data/smollm2-135m", 49152, "smollm2 chatml",
              smol_text, smol_ids, 5, "<|im_start|>", 1);

    const char* llama_text =
        "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>";
    int llama_ids[] = {128006, 882, 128007, 271, 15339, 128009};
    test_real("models/Llama-3.2-1B", 128256, "llama3 header",
              llama_text, llama_ids, 6, "<|start_header_id|>", 128006);

    printf("\n=== added_tokens: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
