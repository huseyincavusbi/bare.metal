/* bench/bench.c -- bare.metal inference benchmark harness.
 *
 * Measures, for one configuration:
 *   - prefill throughput  (prompt processing, bm_forward)
 *   - decode throughput   (bm_step, KV-cache steady state)
 *   - TTFT                (time to first generated token)
 *   - ITL                 (inter-token latency, p50/p90/p99)
 *   - memory              (peak RSS, GPU-allocated bytes, KV bytes/token)
 *   - energy              (optional, when avg watts are supplied by run_bench.sh)
 *
 * Emits a single JSON object (stdout, or --out FILE) so results can be archived
 * and compared across engines/commits.
 *
 * NOTE: bm_forward() currently loops bm_step() internally, so "prefill" is not
 * yet batched; this harness deliberately measures the two phases separately so
 * that gap is visible in the numbers.
 */
#include "baremetal.h"
#include "baremetal/context.h"
#include "baremetal/model.h"
#include "backend/backend.h"
#include "metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char* precision_name(bm_precision_t p) {
    switch (p) {
        case BM_PRECISION_BF16: return "bf16";
        case BM_PRECISION_FP16: return "fp16";
        default:                return "fp32";
    }
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s --model DIR [options]\n"
        "  --prompt-tokens N   prompt length in tokens        (default 128)\n"
        "  --gen M             tokens to generate             (default 128)\n"
        "  --warmup W          discarded warmup reps          (default 2)\n"
        "  --reps R            measured reps                  (default 5)\n"
        "  --temp T            sampling temperature (0=greedy)(default 0)\n"
        "  --seed S            RNG seed                       (default 42)\n"
        "  --precision P       fp32|fp16|bf16 (default device)\n"
        "  --quant q8          quantize matmul weights        (default off)\n"
        "  --avg-w W           average watts (from powermetrics)\n"
        "  --power SRC         ac|battery|unknown             (default unknown)\n"
        "  --out FILE          write JSON here (default stdout)\n",
        prog);
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* out_path  = NULL;
    const char* prec_arg  = NULL;
    const char* power_src = "unknown";
    int   prompt_tokens = 128, gen_tokens = 128, warmup = 2, reps = 5;
    float temp = 0.0f;
    uint64_t seed = 42;
    int   quant = 0;
    double avg_w = 0.0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)         model_dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt-tokens") && i+1 < argc) prompt_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i+1 < argc)           gen_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc)        warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i+1 < argc)          reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)          temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc)          seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc)     prec_arg = argv[++i];
        else if (!strcmp(argv[i], "--quant") && i+1 < argc)         quant = !strcmp(argv[++i], "q8");
        else if (!strcmp(argv[i], "--avg-w") && i+1 < argc)         avg_w = atof(argv[++i]);
        else if (!strcmp(argv[i], "--power") && i+1 < argc)         power_src = argv[++i];
        else if (!strcmp(argv[i], "--out") && i+1 < argc)           out_path = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!model_dir) { usage(argv[0]); return 2; }
    if (prompt_tokens < 1) prompt_tokens = 1;
    if (gen_tokens < 2)    gen_tokens = 2;
    if (reps < 1)          reps = 1;

    /* ---- model ---- */
    double t_load0 = bm_now_ms();
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "bm_create failed\n"); return 1; }

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    if (!m) { fprintf(stderr, "model alloc failed\n"); return 1; }
    bm_load_weights(m, model_dir);
    m->quantized = quant;

    if (prec_arg) {
        if      (!strcmp(prec_arg, "bf16")) m->precision = BM_PRECISION_BF16;
        else if (!strcmp(prec_arg, "fp16")) m->precision = BM_PRECISION_FP16;
        else                                m->precision = BM_PRECISION_FP32;
    } else {
        m->precision = bm_get_supported_precision(ctx);
    }

    bm_compile(m);
    bm_session_t* sess = bm_create_session(ctx, m);
    if (!sess) { fprintf(stderr, "session failed\n"); return 1; }
    double load_ms = bm_now_ms() - t_load0;

    int V = m->arch.vocab_size;
    int max_seq = m->arch.max_seq_len;
    if (prompt_tokens + gen_tokens > max_seq) {
        fprintf(stderr, "warn: prompt+gen (%d+%d) > max_seq (%d); clamping\n",
                prompt_tokens, gen_tokens, max_seq);
        if (prompt_tokens >= max_seq) prompt_tokens = max_seq - 1;
        gen_tokens = max_seq - prompt_tokens;
        if (gen_tokens < 2) gen_tokens = 2;
    }

    /* ---- prompt tokens: encode fixed text, tile to the requested length ---- */
    int* prompt = (int*)malloc((size_t)prompt_tokens * sizeof(int));
    if (!prompt) { fprintf(stderr, "prompt alloc failed\n"); return 1; }
    {
        const char* text =
            "Once upon a time, in a small village at the edge of a great forest, "
            "there lived a curious child who loved to ask questions about the world. "
            "Every morning the child would walk to the river and watch the water flow, "
            "wondering where it came from and where it was going. The villagers said "
            "the river came from the mountains, and the mountains came from the sky, "
            "and the sky held all the answers to every question ever asked.";
        int n_tok = 0;
        int* ids = NULL;
        bm_tokenizer_t* tok = bm_create_tokenizer(model_dir);
        if (tok) { ids = bm_encode(tok, text, &n_tok); }
        if (ids && n_tok > 0) {
            for (int i = 0; i < prompt_tokens; i++) prompt[i] = ids[i % n_tok];
        } else {
            /* fallback: deterministic synthetic ids (keeps the harness usable
             * without a tokenizer present) */
            for (int i = 0; i < prompt_tokens; i++) prompt[i] = (100 + i) % (V > 1 ? V : 2);
        }
        free(ids);
        if (tok) bm_destroy_tokenizer(tok);
    }

    /* ---- measurement buffers ---- */
    double* prefill_ms = calloc((size_t)reps, sizeof(double));
    double* decode_tps = calloc((size_t)reps, sizeof(double));
    double* ttft_ms    = calloc((size_t)reps, sizeof(double));
    double* itl_all    = calloc((size_t)(reps * gen_tokens), sizeof(double));
    if (!prefill_ms || !decode_tps || !ttft_ms || !itl_all) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    int itl_n = 0;
    double active_s = 0.0;
    long   total_gen_tokens = 0;

    /* ---- warmup (discarded: first passes include pipeline work) ---- */
    for (int w = 0; w < warmup; w++) {
        bm_reset_session(sess);
        float* lg_pre = bm_forward(sess, prompt, prompt_tokens);
        bm_sampler_t* smp = bm_create_sampler(V, temp, 1.0f, seed);
        int tk = bm_sample(smp, lg_pre);
        for (int j = 1; j < gen_tokens; j++) {
            float* lg = bm_step(sess, tk);
            tk = bm_sample(smp, lg);
        }
        bm_destroy_sampler(smp);
    }

    /* ---- measured reps ---- */
    for (int r = 0; r < reps; r++) {
        bm_reset_session(sess);
        bm_sampler_t* smp = bm_create_sampler(V, temp, 1.0f, seed);

        double t0 = bm_now_ms();
        float* lg_prefill = bm_forward(sess, prompt, prompt_tokens);
        double t1 = bm_now_ms();
        int tk = bm_sample(smp, lg_prefill);
        double t2 = bm_now_ms();

        prefill_ms[r] = t1 - t0;
        ttft_ms[r]    = t2 - t0;

        double dec_ms = 0.0;
        for (int j = 1; j < gen_tokens; j++) {
            double a = bm_now_ms();
            float* lg = bm_step(sess, tk);
            tk = bm_sample(smp, lg);
            double b = bm_now_ms();
            if (itl_n < reps * gen_tokens) itl_all[itl_n++] = b - a;
            dec_ms += (b - a);
        }
        double t3 = bm_now_ms();

        int decode_steps = gen_tokens - 1;
        decode_tps[r] = dec_ms > 0.0 ? (double)decode_steps / (dec_ms / 1000.0) : 0.0;
        active_s += (t3 - t0) / 1000.0;
        total_gen_tokens += gen_tokens;

        bm_destroy_sampler(smp);
    }

    /* ---- memory ---- */
    size_t rss = bm_peak_rss_bytes();
    size_t gpu = ctx->backend_ctx ? backend_get_allocated_memory(ctx->backend_ctx) : 0;
    double kv_bytes_per_token = (double)m->arch.n_layers * 2.0 * (double)m->kv_dim * 4.0;
    double kv_total = kv_bytes_per_token * (double)max_seq;

    /* ---- aggregate ---- */
    double prefill_tps_med = 0.0;
    if (bm_median_copy(prefill_ms, reps) > 0.0 && prompt_tokens > 0) {
        double med_ms = bm_median_copy(prefill_ms, reps);
        prefill_tps_med = (double)prompt_tokens / (med_ms / 1000.0);
    }

    double energy_joules = 0.0, j_per_token = 0.0;
    if (avg_w > 0.0 && active_s > 0.0) {
        energy_joules = avg_w * active_s;
        if (total_gen_tokens > 0) j_per_token = energy_joules / (double)total_gen_tokens;
    }

    /* ---- host metadata ---- */
    char cpu[128] = "", osver[64] = "", git[64] = "";
    bm_sysctl_str("machdep.cpu.brand_string", cpu, sizeof(cpu));
    bm_sysctl_str("kern.osproductversion", osver, sizeof(osver));
    unsigned long long n_cpu = bm_sysctl_u64("hw.ncpu");
    unsigned long long ram   = bm_sysctl_u64("hw.memsize");
    {
        FILE* p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (p) { if (fgets(git, sizeof(git), p)) { char* nl = strchr(git, '\n'); if (nl) *nl = 0; } pclose(p); }
    }
    char cpu_e[256], osver_e[128], git_e[128];
    bm_json_escape(cpu, cpu_e, sizeof(cpu_e));
    bm_json_escape(osver, osver_e, sizeof(osver_e));
    bm_json_escape(git, git_e, sizeof(git_e));

    /* ---- emit JSON ---- */
    FILE* out = out_path ? fopen(out_path, "w") : stdout;
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema\": \"baremetal.bench/v1\",\n");
    fprintf(out, "  \"meta\": {\n");
    fprintf(out, "    \"engine\": \"bare.metal\",\n");
    fprintf(out, "    \"git_commit\": \"%s\",\n", git_e);
    fprintf(out, "    \"host\": \"%s\",\n", cpu_e);
    fprintf(out, "    \"cpu_count\": %llu,\n", n_cpu);
    fprintf(out, "    \"ram_bytes\": %llu,\n", ram);
    fprintf(out, "    \"macos\": \"%s\",\n", osver_e);
    fprintf(out, "    \"power\": \"%s\",\n", power_src);
    fprintf(out, "    \"model\": \"%s\",\n", model_dir);
    fprintf(out, "    \"params\": %zu,\n", m->n_parameters);
    fprintf(out, "    \"precision\": \"%s\",\n", precision_name(m->precision));
    fprintf(out, "    \"quant\": \"%s\"\n", quant ? "q8" : "none");
    fprintf(out, "  },\n");

    fprintf(out, "  \"config\": {\n");
    fprintf(out, "    \"prompt_tokens\": %d,\n", prompt_tokens);
    fprintf(out, "    \"gen_tokens\": %d,\n", gen_tokens);
    fprintf(out, "    \"warmup\": %d,\n", warmup);
    fprintf(out, "    \"reps\": %d,\n", reps);
    fprintf(out, "    \"temp\": %.3f,\n", temp);
    fprintf(out, "    \"seed\": %llu\n", (unsigned long long)seed);
    fprintf(out, "  },\n");

    fprintf(out, "  \"results\": {\n");
    fprintf(out, "    \"load_ms\": { \"median\": %.3f },\n", load_ms);
    fprintf(out, "    \"prefill\": {\n");
    fprintf(out, "      \"tok_s\": { \"median\": %.2f, \"p10\": %.2f, \"p90\": %.2f },\n",
            prefill_tps_med,
            prompt_tokens / (bm_percentile_copy(prefill_ms, reps, 90.0) / 1000.0),
            prompt_tokens / (bm_percentile_copy(prefill_ms, reps, 10.0) / 1000.0));
    fprintf(out, "      \"ms\": { \"median\": %.3f, \"p10\": %.3f, \"p90\": %.3f }\n",
            bm_median_copy(prefill_ms, reps),
            bm_percentile_copy(prefill_ms, reps, 10.0),
            bm_percentile_copy(prefill_ms, reps, 90.0));
    fprintf(out, "    },\n");
    double dec_p10 = 0, dec_p90 = 0;
    {
        double* tmp = malloc((size_t)reps * sizeof(double));
        for (int i = 0; i < reps; i++) tmp[i] = decode_tps[i];
        dec_p10 = bm_percentile(tmp, reps, 10.0);
        for (int i = 0; i < reps; i++) tmp[i] = decode_tps[i];
        dec_p90 = bm_percentile(tmp, reps, 90.0);
        free(tmp);
    }
    fprintf(out, "    \"decode\": { \"tok_s\": { \"median\": %.2f, \"p10\": %.2f, \"p90\": %.2f } },\n",
            bm_median_copy(decode_tps, reps), dec_p10, dec_p90);
    fprintf(out, "    \"ttft_ms\": { \"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f },\n",
            bm_percentile_copy(ttft_ms, reps, 50.0),
            bm_percentile_copy(ttft_ms, reps, 90.0),
            bm_percentile_copy(ttft_ms, reps, 99.0));
    fprintf(out, "    \"itl_ms\": { \"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f },\n",
            bm_percentile_copy(itl_all, itl_n, 50.0),
            bm_percentile_copy(itl_all, itl_n, 90.0),
            bm_percentile_copy(itl_all, itl_n, 99.0));
    fprintf(out, "    \"memory\": {\n");
    fprintf(out, "      \"rss_peak_bytes\": %zu,\n", rss);
    fprintf(out, "      \"gpu_alloc_bytes\": %zu,\n", gpu);
    fprintf(out, "      \"kv_bytes_per_token\": %.0f,\n", kv_bytes_per_token);
    fprintf(out, "      \"kv_total_bytes\": %.0f\n", kv_total);
    fprintf(out, "    },\n");
    fprintf(out, "    \"energy\": { \"avg_w\": %.2f, \"joules\": %.2f, \"j_per_token\": %.4f },\n",
            avg_w, energy_joules, j_per_token);
    fprintf(out, "    \"active_seconds\": %.3f,\n", active_s);
    fprintf(out, "    \"generated_tokens\": %ld\n", total_gen_tokens);
    fprintf(out, "  }\n");
    fprintf(out, "}\n");

    if (out != stdout) fclose(out);

    free(prefill_ms); free(decode_tps); free(ttft_ms); free(itl_all); free(prompt);
    bm_destroy_session(sess);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return 0;
}
