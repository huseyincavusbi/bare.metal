#include "baremetal.h"
#include "baremetal/lora.h"
#include "baremetal/sft.h"
#include "baremetal/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int* ids;
    int n;
    int cap;
    int stop_id;
} gen_t;

static int collect_token(int token_id, void* user_data) {
    gen_t* g = (gen_t*)user_data;
    if (token_id == g->stop_id) return 1;
    if (g->n < g->cap) g->ids[g->n++] = token_id;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model_dir> <data.jsonl> <adapter_out> <prompt> "
                        "[--steps N] [--seq-len N] [--gen N]\n", argv[0]);
        return 1;
    }
    const char* model_dir = argv[1];
    const char* data_path = argv[2];
    const char* adapter_path = argv[3];
    const char* prompt = argv[4];
    int steps = 1, S = 512, gen = 16;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i + 1 < argc) S = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (steps < 1 || S < 8 || gen < 1) return 1;

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "no Metal context\n"); return 1; }
    bm_model_t* model = bm_load_model(ctx, model_dir);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_dir); bm_destroy(ctx); return 1; }
    bm_tokenizer_t* tok = bm_create_tokenizer(model_dir);
    if (!tok) { fprintf(stderr, "failed to load tokenizer\n"); bm_destroy_model(model); bm_destroy(ctx); return 1; }

    int tmpl = bmt_sft_detect_template(model_dir);
    if (tmpl < 0) { fprintf(stderr, "unknown template\n"); return 1; }
    bmt_sft_dataset_t ds;
    int skipped = 0;
    if (bmt_sft_load_jsonl(data_path, (bmt_sft_template_t)tmpl, 1, &ds, &skipped) != 0 || ds.n == 0) {
        fprintf(stderr, "no usable SFT example in %s\n", data_path);
        return 1;
    }

    int* inputs = malloc((size_t)S * sizeof(int));
    int* targets = malloc((size_t)S * sizeof(int));
    uint8_t* attn = malloc((size_t)S);
    uint8_t* loss = malloc((size_t)S);
    if (!inputs || !targets || !attn || !loss) { fprintf(stderr, "out of memory\n"); return 1; }
    if (bmt_sft_build_example(tok, &ds.items[0], S, inputs, targets, attn, loss) != 0) {
        fprintf(stderr, "failed to build SFT example\n");
        return 1;
    }

    bm_train_config_t cfg = {0};
    cfg.learning_rate = 3e-4f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.grad_clip = 1.0f;
    cfg.grad_accum_steps = 1;
    cfg.seq_len = S;
    cfg.lora_rank = 8;
    cfg.lora_alpha = 16.0f;
    cfg.lora_targets = BMT_LORA_ALL;
    cfg.lora_seed = 1;
    bm_trainer_t* trainer = bm_create_trainer(ctx, model, &cfg);
    if (!trainer) { fprintf(stderr, "failed to create trainer\n"); return 1; }

    for (int i = 0; i < steps; i++) {
        float l = bm_train_step_masked(trainer, inputs, targets, attn, loss, 1, S);
        printf("step %d/%d loss=%.4f\n", i + 1, steps, l);
    }
    bm_train_flush(trainer);
    if (bm_save_adapter(trainer, adapter_path) != 0) {
        fprintf(stderr, "failed to save adapter\n");
        return 1;
    }
    printf("adapter=%s\n", adapter_path);

    bm_destroy_trainer(trainer);
    bm_destroy_model(model);
    bmt_sft_free_dataset(&ds);
    free(inputs); free(targets); free(attn); free(loss);

    char* chat_prompt = NULL;
    int stop_id = -1;
    if (bmt_sft_render_chat((bmt_sft_template_t)tmpl, NULL, prompt, tok, &chat_prompt, &stop_id) != 0) {
        fprintf(stderr, "failed to render chat prompt\n");
        return 1;
    }
    int cap = (int)strlen(chat_prompt) * 4 + 256;
    int* ids = malloc((size_t)cap * sizeof(int));
    int n_ids = 0;
    if (!ids || bm_tokenizer_encode(tok, chat_prompt, 0, 0, ids, &n_ids) != 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    bm_model_t* serve_model = bm_load_model(ctx, model_dir);
    if (!serve_model || bm_load_adapter(serve_model, adapter_path) != 0) {
        fprintf(stderr, "failed to load adapter\n");
        return 1;
    }
    gen_t out = { .ids = malloc((size_t)gen * sizeof(int)), .n = 0, .cap = gen, .stop_id = stop_id };
    if (!out.ids) { fprintf(stderr, "out of memory\n"); return 1; }
    if (bm_run_tokens(ctx, serve_model, ids, n_ids, n_ids + gen, 0.0f, 1, 1.0f, 42,
                       collect_token, &out) != 0) {
        fprintf(stderr, "generation failed\n");
        return 1;
    }

    printf("%s", prompt);
    for (int i = 0; i < out.n; i++) {
        int prev = i > 0 ? out.ids[i - 1] : 1;
        char* piece = bm_tokenizer_decode(tok, prev, out.ids[i]);
        if (piece) bm_tokenizer_safe_print(piece);
    }
    printf("\n");

    free(out.ids);
    free(ids);
    free(chat_prompt);
    bm_destroy_model(serve_model);
    bm_tokenizer_free(tok);
    bm_destroy(ctx);
    return 0;
}
