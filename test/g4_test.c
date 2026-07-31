/* g4_test.c -- Gate 4: loss decreases on real TinyStories data.
 * Reads raw story text, tokenizes with our bm_encode (the real dataloader path),
 * trains SmolLM2-135M for 200 steps (S=8), logs loss every 20 steps.
 * PASS: average loss of last 20 steps < average of first 20 steps. */
#include "baremetal.h"
#include "baremetal/trainer.h"
#include "baremetal/model.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    const char* md = "data/smollm2-135m";
    const int S = 8, NSTEPS = 200, LOG_EVERY = 20;

    /* 1. read raw text */
    FILE* f = fopen("test/grad/g4_text.txt", "r");
    if (!f) { fprintf(stderr, "cannot open g4_text.txt\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* text = malloc(sz + 1); fread(text, 1, sz, f); text[sz] = '\0'; fclose(f);
    fprintf(stderr, "[G4] read %ld chars of TinyStories text\n", sz);

    /* 2. tokenize with our bm_encode (the real C tokenizer) */
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_tokenizer_t* tok = bm_create_tokenizer(md);
    if (!tok) { fprintf(stderr, "tokenizer failed\n"); return 1; }
    int n_tok = 0;
    int* all_tokens = bm_encode(tok, text, &n_tok);
    free(text);
    if (!all_tokens || n_tok < S * (NSTEPS + 1)) {
        fprintf(stderr, "not enough tokens: %d (need %d)\n", n_tok, S*(NSTEPS+1));
        return 1;
    }
    fprintf(stderr, "[G4] tokenized: %d tokens (%.1fK), enough for %d steps\n", n_tok, n_tok/1000.0, n_tok/S - 1);
    fprintf(stderr, "[G4] first8: %d %d %d %d %d %d %d %d\n", all_tokens[0], all_tokens[1], all_tokens[2],
            all_tokens[3], all_tokens[4], all_tokens[5], all_tokens[6], all_tokens[7]);

    /* 3. create model + trainer */
    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, md);
    bm_train_config_t cfg = {.learning_rate=1e-3f,.beta1=0.9f,.beta2=0.999f,.epsilon=1e-8f,
                            .weight_decay=0.0f,.grad_clip=0,.grad_accum_steps=1,.warmup_steps=0,
                            .max_steps=NSTEPS,.use_master_weights=1};
    bm_trainer_t* t = bmt_trainer_create(ctx, m, &cfg, S);

    /* 4. train */
    float losses[NSTEPS];
    int offset = 0;
    printf("step |  loss  | avg(first 20) | avg(last 20)\n");
    printf("-----+--------+----------------+---------------\n");
    for (int i = 0; i < NSTEPS; i++) {
        const int* inp = all_tokens + offset;
        const int* tgt = all_tokens + offset + 1;
        float loss = bm_train_step(t, inp, tgt, 1, S);
        losses[i] = loss;
        offset += S;
        if (i % LOG_EVERY == 0 || i == NSTEPS - 1) {
            float first_avg = 0, last_avg = 0;
            for (int j = 0; j < 20 && j <= i; j++) first_avg += losses[j];
            first_avg /= (i < 20 ? i+1 : 20);
            for (int j = (i >= 19 ? i-19 : 0); j <= i; j++) last_avg += losses[j];
            last_avg /= (i < 20 ? i+1 : 20);
            printf("%4d | %.4f |    %.4f      |    %.4f\n", i, loss, first_avg, last_avg);
        }
    }

    /* 5. check loss decreased */
    float first_avg = 0, last_avg = 0;
    for (int i = 0; i < 20; i++) first_avg += losses[i];
    for (int i = NSTEPS-20; i < NSTEPS; i++) last_avg += losses[i];
    first_avg /= 20; last_avg /= 20;
    printf("\nfirst-20 avg loss: %.4f\n", first_avg);
    printf("last-20  avg loss: %.4f\n", last_avg);
    int ok = last_avg < first_avg;
    printf("\n=== G4 (loss decreases on TinyStories, %d steps): %s (%.4f -> %.4f) ===\n",
           NSTEPS, ok ? "PASS" : "FAIL", first_avg, last_avg);

    free(all_tokens);
    bm_destroy_tokenizer(tok);
    bm_destroy_trainer(t);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return ok ? 0 : 1;
}
