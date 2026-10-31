/* tokenizer_test.c — SentencePiece BPE, Unigram, byte fallback, normalization.
 * Real-tokenizer cases run only when the tokenizer directory exists. */
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
    char out[256];
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

static const char* k_spm_bpe =
    "{"
    " \"model\": {"
    "  \"type\": \"BPE\","
    "  \"byte_fallback\": true,"
    "  \"unk_id\": 0,"
    "  \"vocab\": {"
    "   \"<unk>\": 0, \"<s>\": 1, \"</s>\": 2,"
    "   \"<0xC3>\": 3, \"<0xA9>\": 4,"
    "   \"\xE2\x96\x81\": 5, \"O\": 6, \"n\": 7, \"c\": 8, \"e\": 9,"
    "   \"\xE2\x96\x81O\": 10, \"\xE2\x96\x81On\": 11,"
    "   \"\xE2\x96\x81Onc\": 12, \"\xE2\x96\x81Once\": 13"
    "  },"
    "  \"merges\": [\"\xE2\x96\x81 O\", \"\xE2\x96\x81O n\","
    "               \"\xE2\x96\x81On c\", \"\xE2\x96\x81Onc e\"]"
    " },"
    " \"normalizer\": {\"type\": \"Prepend\", \"prepend\": \"\xE2\x96\x81\"},"
    " \"pre_tokenizer\": null"
    "}";

static const char* k_spm_bpe_no_prepend =
    "{"
    " \"model\": {"
    "  \"type\": \"BPE\","
    "  \"byte_fallback\": true,"
    "  \"unk_id\": 0,"
    "  \"vocab\": {"
    "   \"<unk>\": 0, \"<s>\": 1, \"</s>\": 2,"
    "   \"\xE2\x96\x81\": 5, \"O\": 6, \"n\": 7, \"c\": 8, \"e\": 9,"
    "   \"\xE2\x96\x81O\": 10, \"\xE2\x96\x81On\": 11,"
    "   \"\xE2\x96\x81Onc\": 12, \"\xE2\x96\x81Once\": 13"
    "  },"
    "  \"merges\": [\"\xE2\x96\x81 O\", \"\xE2\x96\x81O n\","
    "               \"\xE2\x96\x81On c\", \"\xE2\x96\x81Onc e\"]"
    " },"
    " \"normalizer\": {\"type\": \"Replace\", \"pattern\": {\"String\": \" \"},"
    "                  \"content\": \"\xE2\x96\x81\"},"
    " \"pre_tokenizer\": {\"type\": \"Split\", \"pattern\": {\"String\": \" \"},"
    "                     \"behavior\": \"MergedWithPrevious\", \"invert\": false}"
    "}";

static const char* k_spm_bpe_array_merges =
    "{\n"
    " \"model\": {\n"
    "  \"type\": \"BPE\",\n"
    "  \"byte_fallback\": true,\n"
    "  \"unk_id\": 0,\n"
    "  \"vocab\": {\n"
    "   \"<unk>\": 0, \"<s>\": 1, \"</s>\": 2,\n"
    "   \"\xE2\x96\x81\": 5, \"O\": 6, \"n\": 7, \"c\": 8, \"e\": 9,\n"
    "   \"\xE2\x96\x81O\": 10, \"\xE2\x96\x81On\": 11,\n"
    "   \"\xE2\x96\x81Onc\": 12, \"\xE2\x96\x81Once\": 13\n"
    "  },\n"
    "  \"merges\": [\n"
    "   [\"\xE2\x96\x81\", \"O\"],\n"
    "   [\"\xE2\x96\x81O\", \"n\"],\n"
    "   [\"\xE2\x96\x81On\", \"c\"],\n"
    "   [\"\xE2\x96\x81Onc\", \"e\"]\n"
    "  ]\n"
    " },\n"
    " \"normalizer\": {\"type\": \"Prepend\", \"prepend\": \"\xE2\x96\x81\"},\n"
    " \"pre_tokenizer\": null\n"
    "}";

static const char* k_unigram =
    "{"
    " \"model\": {"
    "  \"type\": \"Unigram\","
    "  \"unk_id\": 0,"
    "  \"vocab\": ["
    "   [\"<unk>\", 0.0], [\"\xE2\x96\x81\", -1.0], [\"O\", -1.0],"
    "   [\"n\", -1.0], [\"c\", -1.0], [\"e\", -1.0],"
    "   [\"\xE2\x96\x81O\", -0.5], [\"\xE2\x96\x81On\", -0.4],"
    "   [\"\xE2\x96\x81Once\", -0.1]"
    "  ]"
    " },"
    " \"pre_tokenizer\": {\"type\": \"Metaspace\","
    "                     \"replacement\": \"\xE2\x96\x81\","
    "                     \"add_prefix_space\": true, \"split\": true}"
    "}";

static void test_synthetic_spm_bpe(void) {
    char* dir = make_tokenizer_dir(k_spm_bpe);
    check(dir != NULL, "write synthetic SPM-BPE tokenizer");
    if (!dir) return;

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, 14);
    check(t.kind == BM_TOKENIZER_SPM_BPE, "SPM-BPE detected");
    check(t.byte_fallback == 1, "byte fallback detected");

    int want_once[] = {13};
    expect_ids("SPM-BPE Once", &t, "Once", want_once, 1);
    expect_decode("SPM-BPE decode Once", &t, want_once, 1, "Once");

    int want_empty[] = {0};
    expect_ids("SPM-BPE empty", &t, "", want_empty, 0);

    int want_e[] = {5, 3, 4};
    expect_ids("SPM-BPE byte fallback", &t, "\xC3\xA9", want_e, 3);
    expect_decode("SPM-BPE decode bytes", &t, want_e, 3, "\xC3\xA9");

    bm_tokenizer_free(&t);
    free_tokenizer_dir(dir);
}

static void test_synthetic_no_prepend(void) {
    char* dir = make_tokenizer_dir(k_spm_bpe_no_prepend);
    check(dir != NULL, "write no-prepend SPM-BPE tokenizer");
    if (!dir) return;

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, 14);
    check(t.kind == BM_TOKENIZER_SPM_BPE, "no-prepend SPM-BPE detected");
    check(t.prepend_space == 0, "no-prepend detected");

    int want[] = {6, 7, 8, 9};
    expect_ids("no-prepend Once", &t, "Once", want, 4);

    bm_tokenizer_free(&t);
    free_tokenizer_dir(dir);
}

static void test_synthetic_array_merges(void) {
    char* dir = make_tokenizer_dir(k_spm_bpe_array_merges);
    check(dir != NULL, "write array-merges SPM-BPE tokenizer");
    if (!dir) return;

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, 14);
    check(t.kind == BM_TOKENIZER_SPM_BPE, "array-merges SPM-BPE detected");

    int want[] = {13};
    expect_ids("array-merges Once", &t, "Once", want, 1);

    bm_tokenizer_free(&t);
    free_tokenizer_dir(dir);
}

static void test_synthetic_unigram(void) {
    char* dir = make_tokenizer_dir(k_unigram);
    check(dir != NULL, "write synthetic Unigram tokenizer");
    if (!dir) return;

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, 9);
    check(t.kind == BM_TOKENIZER_UNIGRAM, "Unigram detected");

    int want_once[] = {8};
    expect_ids("Unigram Once", &t, "Once", want_once, 1);
    expect_decode("Unigram decode Once", &t, want_once, 1, "Once");

    bm_tokenizer_free(&t);
    free_tokenizer_dir(dir);
}

static void test_real(const char* name, const char* dir, int vocab_size,
                      const int* want, int nwant) {
    char path[600];
    snprintf(path, sizeof(path), "%s/tokenizer.json", dir);
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("SKIP: %s\n", path);
        return;
    }
    fclose(f);

    bm_tokenizer_t t;
    bm_tokenizer_init(&t, dir, vocab_size);
    expect_ids(name, &t, "Once upon a time", want, nwant);
    bm_tokenizer_free(&t);
}

int main(void) {
    test_synthetic_spm_bpe();
    test_synthetic_array_merges();
    test_synthetic_no_prepend();
    test_synthetic_unigram();

    int llama[] = {9038, 2501, 263, 931};
    test_real("TinyLlama", "models/tokenizers/tinyllama", 32000, llama, 4);

    int mistral[] = {6481, 4482, 1032, 1495};
    test_real("Mistral", "models/tokenizers/mistral", 32768, mistral, 4);

    int t5[] = {1447, 1286, 3, 9, 97};
    test_real("T5 Unigram", "models/tokenizers/t5", 32100, t5, 5);

    int gemma2[] = {14326, 3054, 476, 1069};
    test_real("Gemma 2", "models/tokenizers/gemma2", 256000, gemma2, 4);

    int gemma3[] = {14946, 3324, 496, 990};
    test_real("Gemma 3", "models/tokenizers/gemma3", 262144, gemma3, 4);

    int smollm[] = {6403, 1980, 253, 655};
    test_real("SmolLM2 BPE", "models/smollm2-135m", 49152, smollm, 4);

    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all tokenizer tests passed\n");
    return 0;
}
