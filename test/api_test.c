/* api_test.c — exercises ONLY the public baremetal.h API.
 * Verifies the previously-stubbed sampler + tokenizer functions now work:
 *   bm_create_tokenizer / bm_encode / bm_decode / bm_destroy_tokenizer
 *   bm_create_sampler    / bm_sample  / bm_destroy_sampler
 * No internal headers, no model load — pure public-API smoke test. */
#include "baremetal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* model_dir = argc > 1 ? argv[1] : "data/smollm2-135m";
    int failures = 0;

    /* ---- Tokenizer round-trip ---- */
    bm_tokenizer_t* tok = bm_create_tokenizer(model_dir);
    if (!tok) { printf("FAIL: bm_create_tokenizer returned NULL\n"); return 1; }
    printf("PASS: bm_create_tokenizer(%s)\n", model_dir);

    const char* prompt = "Once upon a time";
    int n = 0;
    int* ids = bm_encode(tok, prompt, &n);
    if (!ids || n <= 0) { printf("FAIL: bm_encode returned %p, n=%d\n", (void*)ids, n); failures++; }
    else {
        printf("PASS: bm_encode(\"%s\") -> %d tokens:", prompt, n);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
        /* PyTorch reference for this exact model+prompt is [6403, 1980, 253, 655] */
        int expected[] = {6403, 1980, 253, 655};
        int ok = (n == 4);
        for (int i = 0; i < 4 && ok; i++) ok = (ids[i] == expected[i]);
        printf("%s: token ids match PyTorch [6403,1980,253,655]\n", ok ? "PASS" : "FAIL");
        if (!ok) failures++;

        /* decode round-trip */
        char* piece = bm_decode(tok, 1, ids[0]);
        printf("%s: bm_decode(token %d) = \"%s\" (expect \"Once\")\n",
               (piece && strncmp(piece, "Once", 4) == 0) ? "PASS" : "FAIL", ids[0], piece ? piece : "(null)");
        if (!piece || strncmp(piece, "Once", 4) != 0) failures++;
        free(ids);
    }

    /* ---- Sampler ---- */
    int V = 49152;
    bm_sampler_t* s = bm_create_sampler(V, 0.0f, 0.9f, 42);
    if (!s) { printf("FAIL: bm_create_sampler returned NULL\n"); bm_destroy_tokenizer(tok); return 1; }
    printf("PASS: bm_create_sampler(vocab=%d, temp=0)\n", V);

    /* temp=0 -> greedy argmax. Feed a fake logits vector where index 1234 is max. */
    float* logits = (float*)calloc(V, sizeof(float));
    logits[1234] = 100.0f;
    int pick = bm_sample(s, logits);
    printf("%s: bm_sample(temp=0) -> %d (expect 1234, greedy argmax)\n",
           pick == 1234 ? "PASS" : "FAIL", pick);
    if (pick != 1234) failures++;

    /* temp>0 with a peaked distribution should still usually pick the peak. */
    bm_destroy_sampler(s);
    s = bm_create_sampler(V, 0.7f, 0.9f, 1);
    int hits = 0;
    for (int t = 0; t < 20; t++) { if (bm_sample(s, logits) == 1234) hits++; }
    printf("%s: bm_sample(temp=0.7) picked peak %d/20 times (expect most)\n",
           hits >= 10 ? "PASS" : "FAIL", hits);
    if (hits < 10) failures++;

    free(logits);
    bm_destroy_sampler(s);
    bm_destroy_tokenizer(tok);

    printf("\n=== %s: %d failure(s) ===\n", failures ? "FAIL" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
