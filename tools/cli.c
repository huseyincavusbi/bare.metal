// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal.h"
#include "baremetal/lora.h"
#include "baremetal/sft.h"
#include "baremetal/model.h"
#include "baremetal/tokenizer.h"
#include "baremetal/context.h"
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "utils/log.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mach/mach.h>

extern int bm_run(bm_context_t* ctx, bm_model_t* model, const char* prompt,
                  int steps, float temperature, unsigned long long seed,
                  const char* tok_path);

extern int bm_run_tokens(bm_context_t* ctx, bm_model_t* model,
                         const int* prompt_ids, int n_prompt,
                         int steps, float temperature, int top_k, float top_p,
                         uint64_t seed, bm_token_cb_t callback, void* user_data);

static int load_model_or_fail(bm_model_t* model, const char* path) {
    bm_load_weights(model, path);
    if (!model->weight_buffer) {
        fprintf(stderr, "error: failed to load weights from %s\n", path);
        return -1;
    }
    return 0;
}

#ifdef BAREMETAL_TRAIN
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static size_t get_rss_mb(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size / (1024 * 1024);
    }
    return 0;
}

static size_t get_gpu_memory_mb(bm_context_t* ctx) {
    return backend_get_allocated_memory(ctx->backend_ctx) / (1024 * 1024);
}
#endif

static void print_usage(const char* prog) {
    printf("bare.metal - LLM inference engine for Apple Silicon\n\n");
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  run <model_dir> <prompt> [opts]  Run inference (text in, text out)\n");
    printf("  run-tokens <model_dir> <prompt.bin> <output.bin>  Run with pre-tokenized input\n");
    printf("  tok-test <model_dir> <text>      Test tokenizer encode/decode\n");
    printf("  info <checkpoint>           Print model information\n");
#ifdef BAREMETAL_TRAIN
    printf("  train <model_dir> <data_file> <output_dir>  Train model on text data\n");
    printf("  train-sft <model_dir> <data.jsonl> <output_dir>  Fine-tune on chat data\n");
    printf("    Options:\n");
    printf("      --steps <int>       Max training steps (default: 100)\n");
    printf("      --seq-len <int>     Sequence length S (default: 64)\n");
    printf("      --batch-size <int>  Batch size B (default: 1)\n");
    printf("      --grad-accum <int>  Micro-steps per optimizer step (default: 1)\n");
    printf("      --lr <float>        Learning rate (default: 3e-4)\n");
    printf("      --save-every <int>  Save checkpoint every N steps (default: 0)\n");
    printf("      --resume <path>     Resume from checkpoint\n");
#endif
    printf("  test-dispatch               Test Metal kernel dispatch\n");
    printf("  test-matmul                 Test Metal matmul kernel\n");
    printf("  test-kernels                Test all forward kernels\n");
    printf("\nGlobal Options:\n");
    printf("  -t, --temperature <float>   Sampling temperature (default: 1.0)\n");
    printf("  -p, --topp <float>          Top-p threshold (default: 0.9)\n");
    printf("  -n, --steps <int>           Max generation steps (default: 256)\n");
    printf("  -s, --seed <int>            RNG seed (default: time-based)\n");
    printf("      --quant <q8|q4>         Quantize matmul weights (default: off/fp32)\n");
    printf("      --precision <bf16|fp16|fp32>  Weight precision (default: bf16)\n");
}

static int cmd_test_dispatch(void) {
#define TEST_N 8
    float result[TEST_N];
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "Failed to create backend\n"); return 1; }
    if (!be->library) {
        fprintf(stderr, "No Metal library loaded\n");
        backend_destroy(be);
        return 1;
    }

    backend_kernel_t* kn = backend_kernel_create(be, "vec_add");
    if (!kn) { fprintf(stderr, "vec_add kernel not found\n"); backend_destroy(be); return 1; }

    float a[TEST_N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[TEST_N] = {10, 20, 30, 40, 50, 60, 70, 80};

    backend_buffer_t* bufA = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufB = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufR = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufN = backend_buffer_alloc(be, sizeof(int));

    memcpy(backend_buffer_map(bufA), a, TEST_N * sizeof(float));
    memcpy(backend_buffer_map(bufB), b, TEST_N * sizeof(float));
    int n_val = TEST_N;
    memcpy(backend_buffer_map(bufN), &n_val, sizeof(int));
    backend_buffer_unmap(bufA);
    backend_buffer_unmap(bufB);
    backend_buffer_unmap(bufN);

    backend_buffer_t* bufs[] = {bufA, bufB, bufR, bufN};
    int ok = 1;
    if (backend_kernel_dispatch(be, kn, bufs, NULL, 4,
                               TEST_N, 1, 1,  TEST_N, 1, 1) != 0) {
        fprintf(stderr, "Dispatch failed\n");
        ok = 0;
        goto dispatch_cleanup;
    }

    memcpy(result, backend_buffer_map(bufR), TEST_N * sizeof(float));

    printf("GPU vec_add test: ");
    for (int i = 0; i < TEST_N; i++) {
        printf("%.0f ", result[i]);
        if (result[i] != a[i] + b[i]) ok = 0;
    }
    printf("\n%s\n", ok ? "PASS" : "FAIL");

dispatch_cleanup:
    backend_buffer_free(bufA);
    backend_buffer_free(bufB);
    backend_buffer_free(bufR);
    backend_buffer_free(bufN);
    backend_kernel_destroy(kn);
    backend_destroy(be);
    return ok ? 0 : 1;
}

static int cmd_test_matmul(void) {
    backend_ctx_t* be = backend_create();
    if (!be || !be->library) { fprintf(stderr, "No Metal backend\n"); return 1; }

    backend_kernel_t* kn = backend_kernel_create(be, "matmul_forward_naive");
    if (!kn) { fprintf(stderr, "Kernel not found\n"); backend_destroy(be); return 1; }

    const int BT = 1, C = 4, OC = 3;
    float inp[4] = {1, 2, 3, 4};
    float weight[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    int params[4] = {BT, C, OC, 0}; // has_bias=0

    backend_buffer_t* buf_inp = backend_buffer_alloc(be, BT * C * sizeof(float));
    backend_buffer_t* buf_wgt = backend_buffer_alloc(be, OC * C * sizeof(float));
    backend_buffer_t* buf_bias = backend_buffer_alloc(be, sizeof(float));
    backend_buffer_t* buf_out = backend_buffer_alloc(be, BT * OC * sizeof(float));
    backend_buffer_t* buf_params = backend_buffer_alloc(be, 4 * sizeof(int));

    memcpy(backend_buffer_map(buf_inp), inp, BT * C * sizeof(float));
    memcpy(backend_buffer_map(buf_wgt), weight, OC * C * sizeof(float));
    memcpy(backend_buffer_map(buf_params), params, 4 * sizeof(int));

    backend_buffer_t* bufs[] = {buf_inp, buf_wgt, buf_bias, buf_out, buf_params};
    int ok = 1;
    if (backend_kernel_dispatch(be, kn, bufs, NULL, 5,
                               BT, OC, 1,  1, 1, 1) != 0) {
        fprintf(stderr, "Dispatch failed\n");
        ok = 0;
        goto matmul_cleanup;
    }

    float out[3];
    memcpy(out, backend_buffer_map(buf_out), BT * OC * sizeof(float));
    printf("GPU matmul test: [%.0f %.0f %.0f] ",
           out[0], out[1], out[2]);
    if (out[0]==1 && out[1]==2 && out[2]==3) {
        printf("PASS\n");
    } else {
        printf("FAIL (expected [1 2 3])\n");
        ok = 0;
    }

matmul_cleanup:
    backend_buffer_free(buf_inp);
    backend_buffer_free(buf_wgt);
    backend_buffer_free(buf_bias);
    backend_buffer_free(buf_out);
    backend_buffer_free(buf_params);
    backend_kernel_destroy(kn);
    backend_destroy(be);
    return ok ? 0 : 1;
}

static int cmd_test_kernels(void) {
    backend_ctx_t* be = backend_create();
    if (!be || !be->library) { fprintf(stderr, "No Metal backend\n"); return 1; }
    int failures = 0;

    // --- softmax ---
    {
        backend_kernel_t* kn = backend_kernel_create(be, "softmax_forward");
        int N = 2, C = 3;
        float inp[6] = {1,2,3, 1,2,3};
        float out[6] = {0};
        backend_buffer_t* bufs[3];
        bufs[0] = backend_buffer_alloc(be, N*C*sizeof(float));
        bufs[1] = backend_buffer_alloc(be, N*C*sizeof(float));
        int params[2] = {N, C};
        bufs[2] = backend_buffer_alloc(be, 2*sizeof(int));
        memcpy(backend_buffer_map(bufs[0]), inp, N*C*sizeof(float));
        memcpy(backend_buffer_map(bufs[2]), params, 2*sizeof(int));
        backend_kernel_dispatch(be, kn, bufs, NULL, 3, N, 1, 1, 1, 1, 1);
        memcpy(out, backend_buffer_map(bufs[1]), N*C*sizeof(float));
        float r0 = out[0]+out[1]+out[2], r1 = out[3]+out[4]+out[5];
        printf("Softmax: row0_sum=%.4f row1_sum=%.4f %s\n", r0, r1,
               (fabs(r0-1)<0.001f && fabs(r1-1)<0.001f) ? "PASS" : "FAIL");
        if (fabs(r0-1)>=0.001f) failures++;
        backend_buffer_free(bufs[0]); backend_buffer_free(bufs[1]); backend_buffer_free(bufs[2]);
        backend_kernel_destroy(kn);
    }

    // --- gelu ---
    {
        backend_kernel_t* kn = backend_kernel_create(be, "gelu_forward");
        int N = 4;
        float inp[4] = {-1, 0, 1, 2};
        float out[4];
        backend_buffer_t* bufs[3];
        bufs[0] = backend_buffer_alloc(be, N*sizeof(float));
        bufs[1] = backend_buffer_alloc(be, N*sizeof(float));
        bufs[2] = backend_buffer_alloc(be, sizeof(int));
        memcpy(backend_buffer_map(bufs[0]), inp, N*sizeof(float));
        memcpy(backend_buffer_map(bufs[2]), &N, sizeof(int));
        backend_kernel_dispatch(be, kn, bufs, NULL, 3, N, 1, 1, N, 1, 1);
        memcpy(out, backend_buffer_map(bufs[1]), N*sizeof(float));
        int ok = (out[3] > out[1]) && (out[0] < 0) && (out[1] == 0);
        printf("GELU: [%.3f %.3f %.3f %.3f] %s\n", out[0], out[1], out[2], out[3],
               ok ? "PASS" : "FAIL");
        if (!ok) failures++;
        backend_buffer_free(bufs[0]); backend_buffer_free(bufs[1]); backend_buffer_free(bufs[2]);
        backend_kernel_destroy(kn);
    }

    backend_destroy(be);
    printf("%d failures\n", failures);
    return failures > 0 ? 1 : 0;
}

typedef struct {
    int* ids;
    int  count;
    int  capacity;
} token_collector_t;

static int collect_token(int token_id, void* user_data) {
    token_collector_t* c = (token_collector_t*)user_data;
    if (c->count >= c->capacity) {
        c->capacity = c->capacity ? c->capacity * 2 : 256;
        c->ids = realloc(c->ids, c->capacity * sizeof(int));
    }
    c->ids[c->count++] = token_id;
    return 0;
}

static int read_token_ids(const char* path, int** out_ids) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int n;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (n < 0) { fclose(f); return -1; }
    *out_ids = malloc(n * sizeof(int));
    if (n > 0 && fread(*out_ids, sizeof(int), n, f) != (size_t)n) {
        free(*out_ids); *out_ids = NULL; fclose(f); return -1;
    }
    fclose(f);
    return n;
}

static int write_token_ids(const char* path, const int* ids, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&n, sizeof(int), 1, f);
    if (n > 0) fwrite(ids, sizeof(int), n, f);
    fclose(f);
    return 0;
}

static int cmd_run_tokens(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s run-tokens <checkpoint> <prompt_ids.bin> <output_ids.bin> [options]\n", argv[0]);
        fprintf(stderr, "  --steps <int>       Max generation steps (default: 256)\n");
        fprintf(stderr, "  --temp <float>      Sampling temperature (default: 0.7)\n");
        fprintf(stderr, "  --top-k <int>       Top-k sampling (default: 40)\n");
        fprintf(stderr, "  --top-p <float>     Top-p sampling (default: 0.9)\n");
        fprintf(stderr, "  --seed <int>        RNG seed, 0=time-based (default: 0)\n");
        return 1;
    }
    const char* ckpt_path = argv[2];
    const char* prompt_path = argv[3];
    const char* output_path = argv[4];

    int steps = 256;
    float temp = 0.7f;
    int top_k = 40;
    float top_p = 0.9f;
    uint64_t seed = 0;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
    }

    int* prompt_ids = NULL;
    int n_prompt = read_token_ids(prompt_path, &prompt_ids);
    if (n_prompt < 0) {
        fprintf(stderr, "error: failed to read prompt_ids from %s\n", prompt_path);
        return 1;
    }
    fprintf(stderr, "[run-tokens] loaded %d prompt tokens\n", n_prompt);

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_model_t* model = calloc(1, sizeof(*model));
    if (load_model_or_fail(model, ckpt_path) != 0) {
        free(prompt_ids);
        bm_destroy_model(model);
        bm_destroy(ctx);
        return 1;
    }
    bm_print_model_info(model);

    token_collector_t collector = {NULL, 0, 0};
    int rc = bm_run_tokens(ctx, model, prompt_ids, n_prompt,
                           steps, temp, top_k, top_p, seed,
                           collect_token, &collector);

    if (rc == 0) {
        fprintf(stderr, "[run-tokens] generated %d tokens\n", collector.count);
        write_token_ids(output_path, collector.ids, collector.count);
        fprintf(stderr, "[run-tokens] wrote output to %s\n", output_path);
    }

    free(collector.ids);
    free(prompt_ids);
    bm_destroy_model(model);
    bm_destroy(ctx);
    return rc;
}

static int cmd_tok_test(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s tok-test <model_dir> <text>\n", argv[0]);
        return 1;
    }
    const char* model_dir = argv[2];
    const char* text = argv[3];

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_model_t* model = calloc(1, sizeof(*model));
    if (load_model_or_fail(model, model_dir) != 0) {
        bm_destroy_model(model);
        bm_destroy(ctx);
        return 1;
    }

    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, model_dir, model->arch.vocab_size);

    int tokens[1024];
    int n_tokens = 0;
    bm_tokenizer_encode(&tok, text, 0, 0, tokens, &n_tokens);

    printf("ids:");
    for (int i = 0; i < n_tokens; i++) printf(" %d", tokens[i]);
    printf("\n");

    printf("tokens:");
    for (int i = 0; i < n_tokens; i++) printf(" [%s]", tok.vocab[tokens[i]]);
    printf("\n");

    printf("decoded:");
    for (int i = 0; i < n_tokens; i++) {
        char* d = bm_tokenizer_decode(&tok, i > 0 ? tokens[i-1] : 1, tokens[i]);
        printf("%s", d);
    }
    printf("\n");

    bm_tokenizer_free(&tok);
    bm_destroy_model(model);
    bm_destroy(ctx);
    return 0;
}

#ifdef BAREMETAL_TRAIN
static int cmd_train_sft(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s train-sft <model_dir> <data.jsonl> <output_dir> [options]\n", argv[0]);
        fprintf(stderr, "  --steps <int>       Max optimizer steps (default: 100)\n");
        fprintf(stderr, "  --seq-len <int>     Sequence length S (default: 128)\n");
        fprintf(stderr, "  --batch-size <int>  Sequences per step (default: 1)\n");
        fprintf(stderr, "  --grad-accum <int>  Micro-steps per optimizer step (default: 1)\n");
        fprintf(stderr, "  --lr <float>        Learning rate (default: 1e-4)\n");
        fprintf(stderr, "  --template <name>   chatml|llama3|auto (default: auto)\n");
        fprintf(stderr, "  --val-split <int>   Hold out every Nth example (0 = off)\n");
        fprintf(stderr, "  --max-examples <n>  Cap loaded examples (0 = all)\n");
        fprintf(stderr, "  --seed <int>        Shuffle and LoRA seed (default: 1)\n");
        fprintf(stderr, "  --save-every <int>  Save trainer state every N steps\n");
        fprintf(stderr, "  --save-adapter FILE Save the trained LoRA adapter\n");
        fprintf(stderr, "  --resume <path>     Resume trainer state\n");
        fprintf(stderr, "  --precision <p>     bf16|fp16|fp32\n");
        fprintf(stderr, "  --lora-rank <int>   LoRA rank (default: 8; 0 = full FT)\n");
        fprintf(stderr, "  --lora-alpha <f>    LoRA alpha (default: 16)\n");
        fprintf(stderr, "  --lora-targets <l>  q,k,v,o,gate,up,down|attn|mlp|all\n");
        fprintf(stderr, "  --lora-seed <int>   LoRA init seed (default: 1)\n");
        return 1;
    }
    const char* model_dir = argv[2];
    const char* data_path = argv[3];
    const char* output_dir = argv[4];

    int max_steps = 100, seq_len = 128, batch_size = 1, grad_accum = 1;
    int save_every = 0, max_examples = 0, val_every = 0, seed = 1;
    float lr = 1e-4f;
    const char* prec_str = NULL;
    const char* resume_path = NULL;
    const char* save_adapter = NULL;
    const char* template_name = "auto";
    int lora_rank = 8, lora_seed = 1, lora_targets = BMT_LORA_ALL;
    float lora_alpha = 16.0f;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) max_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-len") == 0 && i+1 < argc) seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--grad-accum") == 0 && i+1 < argc) grad_accum = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--template") == 0 && i+1 < argc) template_name = argv[++i];
        else if (strcmp(argv[i], "--val-split") == 0 && i+1 < argc) val_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-examples") == 0 && i+1 < argc) max_examples = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-every") == 0 && i+1 < argc) save_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-adapter") == 0 && i+1 < argc) save_adapter = argv[++i];
        else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) resume_path = argv[++i];
        else if (strcmp(argv[i], "--precision") == 0 && i+1 < argc) prec_str = argv[++i];
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) lora_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) lora_alpha = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--lora-targets") == 0 && i+1 < argc) lora_targets = bmt_lora_parse_targets(argv[++i]);
        else if (strcmp(argv[i], "--lora-seed") == 0 && i+1 < argc) lora_seed = atoi(argv[++i]);
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }
    if (seq_len < 8) seq_len = 8;
    if (batch_size < 1) batch_size = 1;
    if (grad_accum < 1) grad_accum = 1;
    if (max_steps < 1) max_steps = 1;

    int tmpl;
    if (strcmp(template_name, "auto") == 0) tmpl = bmt_sft_detect_template(model_dir);
    else tmpl = bmt_sft_parse_template(template_name);
    if (tmpl < 0) { fprintf(stderr, "Unknown template: %s\n", template_name); return 1; }

    bmt_sft_dataset_t ds;
    int skipped = 0;
    if (bmt_sft_load_jsonl(data_path, (bmt_sft_template_t)tmpl, max_examples, &ds, &skipped) != 0) {
        fprintf(stderr, "No usable SFT examples in %s (skipped %d)\n", data_path, skipped);
        return 1;
    }
    fprintf(stderr, "[sft] template=%s examples=%d skipped=%d\n",
            tmpl == BMT_SFT_LLAMA3 ? "llama3" : "chatml", ds.n, skipped);

    int* train_idx = malloc((size_t)ds.n * sizeof(int));
    int* val_idx = malloc((size_t)ds.n * sizeof(int));
    int n_train = 0, n_val = 0;
    for (int i = 0; i < ds.n; i++) {
        if (val_every > 0 && (i % val_every) == 0) val_idx[n_val++] = i;
        else train_idx[n_train++] = i;
    }
    if (n_train == 0) {
        fprintf(stderr, "No training examples after validation split\n");
        free(train_idx); free(val_idx); bmt_sft_free_dataset(&ds);
        return 1;
    }
    unsigned int rng = (unsigned int)seed;
    for (int i = n_train - 1; i > 0; i--) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        int j = (int)(rng % (unsigned int)(i + 1));
        int tmp = train_idx[i]; train_idx[i] = train_idx[j]; train_idx[j] = tmp;
    }

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_model_t* model = calloc(1, sizeof(*model));
    if (load_model_or_fail(model, model_dir) != 0) {
        bm_destroy_model(model); bm_destroy(ctx);
        free(train_idx); free(val_idx); bmt_sft_free_dataset(&ds);
        return 1;
    }
    if (prec_str) {
        if (strcmp(prec_str, "bf16") == 0) model->precision = BM_PRECISION_BF16;
        else if (strcmp(prec_str, "fp16") == 0) model->precision = BM_PRECISION_FP16;
        else model->precision = BM_PRECISION_FP32;
    } else {
        model->precision = bm_get_supported_precision(ctx);
    }
    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, model_dir, model->arch.vocab_size);

    bm_train_config_t cfg = {0};
    cfg.learning_rate = lr;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.grad_accum_steps = grad_accum;
    cfg.warmup_steps = 0;
    cfg.max_steps = max_steps;
    cfg.use_master_weights = 1;
    cfg.seq_len = seq_len;
    cfg.lora_rank = lora_rank;
    cfg.lora_alpha = lora_alpha;
    cfg.lora_targets = lora_targets;
    cfg.lora_seed = lora_seed;

    bm_trainer_t* trainer = bm_create_trainer(ctx, model, &cfg);
    if (!trainer) {
        fprintf(stderr, "Failed to create trainer\n");
        bm_tokenizer_free(&tok); bm_destroy_model(model); bm_destroy(ctx);
        free(train_idx); free(val_idx); bmt_sft_free_dataset(&ds);
        return 1;
    }
    if (resume_path) {
        bm_load_state(trainer, resume_path);
        fprintf(stderr, "[sft] resumed from %s\n", resume_path);
    }

    int bt = batch_size * seq_len;
    int* inputs = malloc((size_t)bt * sizeof(int));
    int* targets = malloc((size_t)bt * sizeof(int));
    uint8_t* attn = malloc((size_t)bt);
    uint8_t* loss = malloc((size_t)bt);
    double total_ms = 0.0;
    long total_tokens = 0;
    int pos = 0;

    for (int step = 0; step < max_steps; step++) {
        int built = 1;
        for (int b = 0; b < batch_size; b++) {
            int ei = train_idx[(pos + b) % n_train];
            if (bmt_sft_build_example(&tok, &ds.items[ei], seq_len,
                                      inputs + b * seq_len, targets + b * seq_len,
                                      attn + b * seq_len, loss + b * seq_len) != 0) {
                built = 0;
                break;
            }
        }
        if (!built) {
            pos += batch_size;
            continue;
        }
        double t0 = get_time_ms();
        float l = bm_train_step_masked(trainer, inputs, targets, attn, loss, batch_size, seq_len);
        double ms = get_time_ms() - t0;
        total_ms += ms;
        total_tokens += (long)batch_size * seq_len;
        printf("step %d/%d loss=%.4f step_ms=%.0f tok/s=%.1f rss=%zuMB\n",
               step + 1, max_steps, l, ms,
               (double)(batch_size * seq_len) * 1000.0 / (ms > 0 ? ms : 1),
               get_rss_mb());
        if (save_every > 0 && (step + 1) % save_every == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/checkpoint_%d.bin", output_dir, step + 1);
            bm_save_state(trainer, path);
        }
        pos += batch_size;
    }
    bm_train_flush(trainer);

    if (save_adapter) {
        if (bm_save_adapter(trainer, save_adapter) == 0)
            fprintf(stderr, "[sft] saved adapter to %s\n", save_adapter);
        else
            fprintf(stderr, "[sft] failed to save adapter to %s\n", save_adapter);
    }

    if (n_val > 0) {
        double vloss = 0.0;
        int vn = 0;
        for (int i = 0; i < n_val; i++) {
            const bmt_sft_pair_t* p = &ds.items[val_idx[i]];
            if (bmt_sft_build_example(&tok, p, seq_len, inputs, targets, attn, loss) != 0)
                continue;
            vloss += bm_train_eval_masked(trainer, inputs, targets, attn, loss, 1, seq_len);
            vn++;
        }
        if (vn > 0) printf("val loss: %.4f (%d examples)\n", vloss / vn, vn);
    }

    printf("avg step: %.1fms, %.1f tok/s, %ld tokens\n",
           total_ms / max_steps,
           (double)total_tokens * 1000.0 / (total_ms > 0 ? total_ms : 1),
           total_tokens);

    free(inputs); free(targets); free(attn); free(loss);
    free(train_idx); free(val_idx);
    bmt_sft_free_dataset(&ds);
    bm_tokenizer_free(&tok);
    bm_destroy_trainer(trainer);
    bm_destroy_model(model);
    bm_destroy(ctx);
    return 0;
}

static int cmd_train(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s train <model_dir> <data_file> <output_dir> [options]\n", argv[0]);
        fprintf(stderr, "  --steps <int>       Max training steps (default: 100)\n");
        fprintf(stderr, "  --seq-len <int>     Sequence length S (default: 64)\n");
        fprintf(stderr, "  --batch-size <int>  Batch size B (default: 1)\n");
        fprintf(stderr, "  --grad-accum <int>  Micro-steps per optimizer step (default: 1)\n");
        fprintf(stderr, "  --lr <float>        Learning rate (default: 3e-4)\n");
        fprintf(stderr, "  --save-every <int>  Save checkpoint every N steps (default: 0 = no save)\n");
        fprintf(stderr, "  --resume <path>     Resume from checkpoint\n");
        fprintf(stderr, "  --precision <bf16|fp16|fp32>  Weight precision (default: bf16)\n");
        fprintf(stderr, "  --lora-rank <int>   LoRA rank (0 = full fine-tune, default: 0)\n");
        fprintf(stderr, "  --lora-alpha <float> LoRA alpha (default: 16)\n");
        fprintf(stderr, "  --lora-targets <list> q,k,v,o,gate,up,down|attn|mlp|all\n");
        fprintf(stderr, "  --lora-seed <int>   LoRA init seed (default: 1)\n");
        fprintf(stderr, "  --save-adapter FILE Save the trained LoRA adapter\n");
        return 1;
    }
    const char* model_dir = argv[2];
    const char* data_file = argv[3];
    const char* output_dir = argv[4];

    int max_steps = 100;
    int seq_len = 64;
    int batch_size = 1;
    int grad_accum = 1;
    float lr = 3e-4f;
    int save_every = 0;
    const char* resume_path = NULL;
    const char* prec_str = NULL;
    int lora_rank = 0, lora_seed = 1, lora_targets = BMT_LORA_ALL;
    float lora_alpha = 16.0f;
    const char* save_adapter = NULL;

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) max_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-len") == 0 && i+1 < argc) seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--grad-accum") == 0 && i+1 < argc) grad_accum = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--save-every") == 0 && i+1 < argc) save_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) resume_path = argv[++i];
        else if (strcmp(argv[i], "--precision") == 0 && i+1 < argc) prec_str = argv[++i];
        else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) lora_rank = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lora-alpha") == 0 && i+1 < argc) lora_alpha = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--lora-targets") == 0 && i+1 < argc) lora_targets = bmt_lora_parse_targets(argv[++i]);
        else if (strcmp(argv[i], "--lora-seed") == 0 && i+1 < argc) lora_seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save-adapter") == 0 && i+1 < argc) save_adapter = argv[++i];
    }

    if (batch_size < 1) {
        fprintf(stderr, "Error: batch-size must be >= 1\n");
        return 1;
    }

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    bm_model_t* model = calloc(1, sizeof(*model));
    if (load_model_or_fail(model, model_dir) != 0) {
        bm_destroy_model(model);
        bm_destroy(ctx);
        return 1;
    }
    bm_print_model_info(model);

    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, model_dir, model->arch.vocab_size);

    FILE* f = fopen(data_file, "r");
    if (!f) {
        fprintf(stderr, "Cannot open data file: %s\n", data_file);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = 0;
    fclose(f);

    int* tokens = malloc(fsize * sizeof(int));
    int n_tokens = 0;
    bm_tokenizer_encode(&tok, text, 0, 0, tokens, &n_tokens);
    free(text);
    fprintf(stderr, "[train] loaded %d tokens from %s\n", n_tokens, data_file);

    bm_train_config_t cfg = {0};
    cfg.learning_rate = lr;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.grad_clip = 1.0f;
    cfg.grad_accum_steps = grad_accum > 0 ? grad_accum : 1;
    cfg.warmup_steps = 10;
    cfg.max_steps = max_steps;
    cfg.seq_len = seq_len;
    cfg.lora_rank = lora_rank;
    cfg.lora_alpha = lora_alpha;
    cfg.lora_targets = lora_targets;
    cfg.lora_seed = lora_seed;

    if (prec_str) {
        if (strcmp(prec_str, "bf16") == 0) model->precision = BM_PRECISION_BF16;
        else if (strcmp(prec_str, "fp16") == 0) model->precision = BM_PRECISION_FP16;
        else model->precision = BM_PRECISION_FP32;
    } else {
        model->precision = bm_get_supported_precision(ctx);
    }

    bm_trainer_t* trainer = bm_create_trainer(ctx, model, &cfg);
    if (!trainer) {
        fprintf(stderr, "Failed to create trainer\n");
        return 1;
    }

    if (resume_path) {
        bm_load_state(trainer, resume_path);
        fprintf(stderr, "[train] resumed from %s\n", resume_path);
    }

    int batch_tokens = batch_size * seq_len;
    int* inputs = malloc(batch_tokens * sizeof(int));
    int* targets = malloc(batch_tokens * sizeof(int));
    int pos = 0;
    double total_step_time = 0.0;

    for (int step = 0; step < max_steps; step++) {
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                inputs[b * seq_len + i] = tokens[pos % n_tokens];
                targets[b * seq_len + i] = tokens[(pos + 1) % n_tokens];
                pos++;
            }
        }

        double step_start = get_time_ms();
        float loss = bm_train_step(trainer, inputs, targets, batch_size, seq_len);
        double step_ms = get_time_ms() - step_start;
        total_step_time += step_ms;

        size_t gpu_mem = get_gpu_memory_mb(ctx);
        size_t rss = get_rss_mb();
        printf("step %d/%d: loss=%.4f  step_ms=%.0f  gpu_mem=%.1fMB  rss=%zuMB\n",
               step + 1, max_steps, loss, step_ms, (double)gpu_mem, rss);

        if (save_every > 0 && (step + 1) % save_every == 0) {
            char path[256];
            snprintf(path, sizeof(path), "%s/checkpoint_%d.bin", output_dir, step + 1);
            bm_save_state(trainer, path);
            fprintf(stderr, "[train] saved checkpoint to %s\n", path);
        }
    }

    bm_train_flush(trainer);
    if (save_adapter) {
        if (bm_save_adapter(trainer, save_adapter) == 0)
            fprintf(stderr, "[train] saved adapter to %s\n", save_adapter);
        else
            fprintf(stderr, "[train] failed to save adapter to %s\n", save_adapter);
    }
    printf("Avg step time: %.1fms\n", total_step_time / max_steps);

    free(inputs);
    free(targets);
    free(tokens);
    bm_tokenizer_free(&tok);
    bm_destroy_trainer(trainer);
    bm_destroy_model(model);
    bm_destroy(ctx);
    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    const char* cmd = argv[1];

    if (strcmp(cmd, "test-dispatch") == 0) return cmd_test_dispatch();
    if (strcmp(cmd, "test-matmul") == 0) return cmd_test_matmul();
    if (strcmp(cmd, "test-kernels") == 0) return cmd_test_kernels();
    if (strcmp(cmd, "run-tokens") == 0) return cmd_run_tokens(argc, argv);
    if (strcmp(cmd, "tok-test") == 0) return cmd_tok_test(argc, argv);
#ifdef BAREMETAL_TRAIN
    if (strcmp(cmd, "train") == 0) return cmd_train(argc, argv);
    if (strcmp(cmd, "train-sft") == 0) return cmd_train_sft(argc, argv);
#endif

    if (strcmp(cmd, "info") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
        bm_model_t* model = calloc(1, sizeof(*model));
        if (load_model_or_fail(model, argv[2]) != 0) {
            bm_destroy_model(model);
            bm_destroy(ctx);
            return 1;
        }
        bm_print_model_info(model);
        bm_destroy_model(model);
        bm_destroy(ctx);
        return 0;
    }

    if (strcmp(cmd, "run") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        const char* model_dir = argv[2];
        const char* prompt = "";
        int steps = 256;
        float temp = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;
        uint64_t seed = 0;
        int quant = 0;   /* 0 = off, 8 = Q8, 4 = Q4 */
        const char* prec_str = NULL;
        const char* adapter_path = NULL;
        int max_seq = 8192;

        for (int i = 3; i < argc; i++) {
            if (argv[i][0] == '-' && argv[i][1] == '-') {
                if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) steps = atoi(argv[++i]);
                else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) temp = (float)atof(argv[++i]);
                else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
                else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc) top_p = (float)atof(argv[++i]);
                else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
                else if (strcmp(argv[i], "--quant") == 0 && i+1 < argc) { const char* q = argv[++i]; quant = (strcmp(q, "q4") == 0) ? 4 : (strcmp(q, "q8") == 0 ? 8 : 0); }
                else if (strcmp(argv[i], "--precision") == 0 && i+1 < argc) prec_str = argv[++i];
                else if (strcmp(argv[i], "--adapter") == 0 && i+1 < argc) adapter_path = argv[++i];
                else if (strcmp(argv[i], "--max-seq") == 0 && i+1 < argc) max_seq = atoi(argv[++i]);
                else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
            } else if (prompt[0] == '\0') {
                prompt = argv[i];
            }
        }

        if (prompt[0] == '\0') {
            fprintf(stderr, "Usage: %s run <model_dir> <prompt> [options]\n", argv[0]);
            return 1;
        }

        bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
        bm_model_t* model = calloc(1, sizeof(*model));
        if (load_model_or_fail(model, model_dir) != 0) {
            bm_destroy_model(model);
            bm_destroy(ctx);
            return 1;
        }
        model->quant_bits = quant;
        if (prec_str) {
            if (strcmp(prec_str, "bf16") == 0) model->precision = BM_PRECISION_BF16;
            else if (strcmp(prec_str, "fp16") == 0) model->precision = BM_PRECISION_FP16;
            else model->precision = BM_PRECISION_FP32;
        } else {
            model->precision = bm_get_supported_precision(ctx);
        }
        if (max_seq > 0 && model->arch.max_seq_len > max_seq)
            model->arch.max_seq_len = max_seq;
        if (adapter_path && bm_load_adapter(model, adapter_path) != 0) {
            fprintf(stderr, "Failed to load adapter: %s\n", adapter_path);
            bm_destroy_model(model);
            bm_destroy(ctx);
            return 1;
        }
        bm_print_model_info(model);

        bm_tokenizer_t tok;
        bm_tokenizer_init(&tok, model_dir, model->arch.vocab_size);

        int ptok[1024];
        int nt = 0;
        bm_tokenizer_encode(&tok, prompt, 0, 0, ptok, &nt);
        fprintf(stderr, "[run] prompt: %d tokens\n", nt);

        printf("%s", prompt);
        fflush(stdout);

        token_collector_t collector = {NULL, 0, 0};
        int rc = bm_run_tokens(ctx, model, ptok, nt,
                               steps, temp, top_k, top_p, seed,
                               collect_token, &collector);

        if (rc == 0) {
            for (int i = 0; i < collector.count; i++) {
                int prev = i > 0 ? collector.ids[i-1] : 1;
                char* piece = bm_tokenizer_decode(&tok, prev, collector.ids[i]);
                bm_tokenizer_safe_print(piece);
            }
            printf("\n");
            fflush(stdout);
        }

        free(collector.ids);
        bm_tokenizer_free(&tok);
        bm_destroy_model(model);
        bm_destroy(ctx);
        return rc;
    }

    print_usage(argv[0]);
    return 1;
}
