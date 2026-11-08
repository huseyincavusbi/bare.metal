/* sft_test.c -- chat templates, JSONL parsing, masks, and an SFT step. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/tokenizer.h"
#include "baremetal/trainer.h"
#include "baremetal/sft.h"
#include "baremetal/lora.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static int write_jsonl(const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"}]}\n", f);
    fputs("{\"prompt\":\"2+2?\",\"completion\":\"4\"}\n", f);
    fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"no assistant\"}]}\n", f);
    fclose(f);
    return 0;
}

int main(void) {
    char path[] = "/tmp/bm_sft_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { fprintf(stderr, "mkstemp failed\n"); return 1; }
    close(fd);
    write_jsonl(path);

    int skipped = 0;
    bmt_sft_dataset_t ds;
    check(bmt_sft_load_jsonl(path, BMT_SFT_CHATML, 0, &ds, &skipped) == 0, "load JSONL");
    check(ds.n == 2, "two valid examples");
    check(skipped == 1, "one malformed example skipped");
    check(strstr(ds.items[0].prompt, "<|im_start|>user\nhi<|im_end|>") != NULL,
          "ChatML user prompt");
    check(strcmp(ds.items[0].completion, "hello<|im_end|>\n") == 0, "ChatML completion");

    bmt_sft_dataset_t dl;
    check(bmt_sft_load_jsonl(path, BMT_SFT_LLAMA3, 0, &dl, NULL) == 0, "load as Llama 3");
    check(strncmp(dl.items[0].prompt, "<|begin_of_text|><|start_header_id|>user",
              strlen("<|begin_of_text|><|start_header_id|>user")) == 0,
          "Llama 3 BOS and header");
    check(strstr(dl.items[0].prompt, "<|start_header_id|>assistant<|end_header_id|>\n\n") != NULL,
          "Llama 3 assistant header");
    check(strcmp(dl.items[0].completion, "hello<|eot_id|>") == 0, "Llama 3 completion");

    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, "data/smollm2-135m", 49152);
    check(bm_tokenizer_token_id(&tok, "<|im_start|>") == 1, "tokenizer special lookup");
    char* chat_prompt = NULL;
    int stop_id = -1;
    check(bmt_sft_render_chat(BMT_SFT_CHATML, NULL, "hi", &tok, &chat_prompt, &stop_id) == 0,
          "render ChatML chat prompt");
    check(strcmp(chat_prompt, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n") == 0,
          "ChatML chat prompt text");
    check(stop_id == 2, "ChatML stop token");
    free(chat_prompt);
    const int S = 64;
    int inputs[64], targets[64];
    uint8_t attn[64], loss[64];
    check(bmt_sft_build_example(&tok, &ds.items[0], S, inputs, targets, attn, loss) == 0,
          "build ChatML example");
    int n_attn = 0, n_loss = 0;
    for (int i = 0; i < S; i++) { n_attn += attn[i]; n_loss += loss[i]; }
    int pn = 0;
    int* pt = malloc((strlen(ds.items[0].prompt) * 4 + 256) * sizeof(int));
    bm_tokenizer_encode(&tok, ds.items[0].prompt, 0, 0, pt, &pn);
    free(pt);
    check(n_loss > 0 && n_loss == n_attn - pn, "prompt and completion mask lengths");
    check(loss[0] == 0, "prompt token has no loss");
    check(loss[n_attn - 2] == 1 && loss[n_attn - 1] == 0,
          "last completion target has loss");
    int shift_ok = 1;
    for (int i = 0; i < S - 1; i++) if (targets[i] != inputs[i + 1]) shift_ok = 0;
    check(shift_ok, "targets are shifted inputs");

    bmt_sft_dataset_t d3;
    check(bmt_sft_load_jsonl(path, BMT_SFT_CHATML, 0, &d3, NULL) == 0, "reload for trainer");
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, "data/smollm2-135m");
    m->precision = bm_get_supported_precision(ctx);
    bm_train_config_t cfg = {0};
    cfg.learning_rate = 1e-4f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.grad_accum_steps = 1;
    cfg.use_master_weights = 1;
    cfg.lora_rank = 8;
    cfg.lora_alpha = 16.0f;
    cfg.lora_targets = BMT_LORA_Q;
    cfg.lora_seed = 1;
    bm_trainer_t* t = bmt_trainer_create(ctx, m, &cfg, S);
    check(t != NULL, "create SFT trainer");
    if (t) {
        float last = 0.0f;
        int finite = 1;
        for (int step = 0; step < 5; step++) {
            const bmt_sft_pair_t* p = &d3.items[step % d3.n];
            bmt_sft_build_example(&tok, p, S, inputs, targets, attn, loss);
            last = bm_train_step_masked(t, inputs, targets, attn, loss, 1, S);
            if (!isfinite(last)) finite = 0;
        }
        printf("SFT step loss: %.6f\n", last);
        check(finite, "SFT steps produce finite loss");
        bm_destroy_trainer(t);
    }
    bm_destroy_model(m);
    bm_destroy(ctx);

    bmt_sft_free_dataset(&ds);
    bmt_sft_free_dataset(&dl);
    bmt_sft_free_dataset(&d3);
    bm_tokenizer_free(&tok);
    remove(path);

    printf("\n=== sft: %s ===\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
