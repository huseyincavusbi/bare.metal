/* parity.c — compare bare.metal against a transformers golden fixture.
 *
 * Fixture format: baremetal.parity/v1 (see test/parity/make_fixture.py).
 * Checks: full-vocabulary logits cosine, argmax, and greedy token ids.
 */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/json.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static double cosine(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static char* read_text(const char* path) {
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

static float* read_floats(const char* path, int expected) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open logits file: %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long bytes = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    if (bytes != (long)expected * (long)sizeof(float)) {
        fprintf(stderr, "logits size mismatch: %ld bytes, expected %ld\n",
                bytes, (long)expected * (long)sizeof(float));
        fclose(f);
        return NULL;
    }
    float* buf = malloc((size_t)bytes);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)bytes, f) != (size_t)bytes) {
        fprintf(stderr, "short read: %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return buf;
}

static int* read_int_array(const bmt_json_t* arr, int* count) {
    *count = bmt_json_array_count(arr);
    if (*count <= 0) return NULL;
    int* out = malloc((size_t)(*count) * sizeof(int));
    if (!out) return NULL;
    for (int i = 0; i < *count; i++) {
        if (bmt_json_int(bmt_json_array_get(arr, i), &out[i]) != 0) {
            free(out);
            return NULL;
        }
    }
    return out;
}

static void fixture_dir(char* out, size_t out_len, const char* fixture) {
    const char* slash = strrchr(fixture, '/');
    if (!slash) {
        snprintf(out, out_len, ".");
        return;
    }
    size_t n = (size_t)(slash - fixture);
    if (n >= out_len) n = out_len - 1;
    memcpy(out, fixture, n);
    out[n] = '\0';
}

static bm_precision_t parse_precision(const char* s) {
    if (strcmp(s, "bf16") == 0) return BM_PRECISION_BF16;
    if (strcmp(s, "fp16") == 0) return BM_PRECISION_FP16;
    return BM_PRECISION_FP32;
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* fixture_path = NULL;
    const char* prec_arg = NULL;
    int max_gen = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) fixture_path = argv[++i];
        else if (strcmp(argv[i], "--precision") == 0 && i + 1 < argc) prec_arg = argv[++i];
        else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc) max_gen = atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s --model DIR --fixture FILE [--precision fp32|fp16|bf16] [--gen N]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!model_dir || !fixture_path) {
        fprintf(stderr, "usage: %s --model DIR --fixture FILE [--precision P] [--gen N]\n", argv[0]);
        return 2;
    }

    char* text = read_text(fixture_path);
    if (!text) {
        fprintf(stderr, "cannot read fixture: %s\n", fixture_path);
        return 1;
    }
    char err[160];
    bmt_json_t* fx = bmt_json_parse(text, err, sizeof(err));
    free(text);
    if (!fx) {
        fprintf(stderr, "invalid fixture JSON: %s\n", err);
        return 1;
    }

    const char* prec_name = bmt_json_string(bmt_json_get(fx, "precision"));
    if (prec_arg) prec_name = prec_arg;
    if (!prec_name) prec_name = "fp32";
    bm_precision_t precision = parse_precision(prec_name);

    int np = 0, ng = 0;
    int* prompt = read_int_array(bmt_json_get(fx, "prompt_token_ids"), &np);
    int* golden_gen = read_int_array(bmt_json_get(fx, "generated_token_ids"), &ng);
    if (!prompt || !golden_gen) {
        fprintf(stderr, "fixture missing prompt_token_ids or generated_token_ids\n");
        bmt_json_free(fx);
        free(prompt);
        free(golden_gen);
        return 1;
    }
    if (max_gen > 0 && max_gen < ng) ng = max_gen;

    int vocab = 0;
    bmt_json_int(bmt_json_get(fx, "vocab_size"), &vocab);
    const char* logits_file = bmt_json_string(bmt_json_get(fx, "logits_file"));
    if (vocab <= 0 || !logits_file) {
        fprintf(stderr, "fixture missing vocab_size or logits_file\n");
        bmt_json_free(fx);
        free(prompt);
        free(golden_gen);
        return 1;
    }

    char dir[1024], logits_path[2048];
    fixture_dir(dir, sizeof(dir), fixture_path);
    snprintf(logits_path, sizeof(logits_path), "%s/%s", dir, logits_file);
    float* golden_logits = read_floats(logits_path, vocab);
    if (!golden_logits) {
        bmt_json_free(fx);
        free(prompt);
        free(golden_gen);
        return 1;
    }

    const char* model_name = bmt_json_string(bmt_json_get(fx, "model"));
    printf("parity: %s (%s)\n", model_name ? model_name : model_dir, prec_name);
    printf("  prompt: %d tokens, gen: %d\n", np, ng);

    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) {
        fprintf(stderr, "bm_create failed\n");
        return 1;
    }
    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    if (!m) {
        fprintf(stderr, "model alloc failed\n");
        bm_destroy(ctx);
        return 1;
    }
    bm_load_weights(m, model_dir);
    if (!m->weight_buffer) {
        fprintf(stderr, "failed to load weights from %s\n", model_dir);
        free(m);
        bm_destroy(ctx);
        return 1;
    }
    m->precision = precision;
    bm_compile(m);
    bm_session_t* sess = bm_create_session(ctx, m);
    if (!sess) {
        fprintf(stderr, "session creation failed\n");
        bm_destroy_model(m);
        bm_destroy(ctx);
        return 1;
    }
    if (vocab != m->arch.vocab_size) {
        fprintf(stderr, "vocab mismatch: fixture %d, model %d\n",
                vocab, m->arch.vocab_size);
        return 1;
    }

    bm_reset_session(sess);
    float* logits = bm_forward(sess, prompt, np);
    if (!logits) {
        fprintf(stderr, "forward failed\n");
        return 1;
    }

    double cos = cosine(logits, golden_logits, vocab);
    double max_diff = 0.0;
    for (int i = 0; i < vocab; i++) {
        double d = fabs((double)logits[i] - (double)golden_logits[i]);
        if (d > max_diff) max_diff = d;
    }
    int got_argmax = argmax(logits, vocab);
    int want_argmax = argmax(golden_logits, vocab);
    double threshold = precision == BM_PRECISION_FP32 ? 0.9999 : 0.999;

    int failures = 0;
    printf("  logits cos: %.7f (min %.4f)  max|diff|: %.3e\n", cos, threshold, max_diff);
    if (cos < threshold) {
        printf("FAIL: logits cosine below threshold\n");
        failures++;
    }
    printf("  argmax: got %d, want %d\n", got_argmax, want_argmax);
    if (got_argmax != want_argmax) {
        printf("FAIL: argmax mismatch\n");
        failures++;
    }

    int* gen = malloc((size_t)ng * sizeof(int));
    if (!gen) return 1;
    gen[0] = argmax(logits, vocab);
    for (int i = 1; i < ng; i++) {
        logits = bm_step(sess, gen[i - 1]);
        if (!logits) {
            printf("FAIL: step %d returned NULL\n", i);
            failures++;
            break;
        }
        gen[i] = argmax(logits, vocab);
    }

    int first_bad = -1;
    for (int i = 0; i < ng; i++) {
        if (gen[i] != golden_gen[i]) {
            first_bad = i;
            break;
        }
    }
    if (first_bad < 0) {
        printf("  tokens: match (%d/%d)\n", ng, ng);
    } else {
        printf("FAIL: token mismatch at %d: got %d, want %d\n",
               first_bad, gen[first_bad], golden_gen[first_bad]);
        failures++;
    }

    free(gen);
    free(golden_logits);
    free(golden_gen);
    free(prompt);
    bmt_json_free(fx);
    bm_destroy_session(sess);
    bm_destroy_model(m);
    bm_destroy(ctx);

    if (failures) {
        printf("PARITY FAIL\n");
        return 1;
    }
    printf("PARITY PASS\n");
    return 0;
}
