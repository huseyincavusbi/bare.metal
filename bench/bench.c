/* bench/bench.c -- bare.metal inference benchmark harness.
 *
 * Measures, for one configuration:
 *   - prefill throughput  (prompt processing, bm_forward)
 *   - decode throughput   (bm_step, KV-cache steady state)
 *   - TTFT                (time to first generated token)
 *   - ITL                 (inter-token latency, full distribution)
 *   - memory              (peak RSS, GPU-allocated bytes, KV bytes/token)
 *   - energy              (when avg watts are supplied by run_bench.sh)
 *
 * Full statistics (n/min/max/mean/stddev/p50/p90/p99) plus raw samples and the
 * generated token ids (greedy) are emitted so runs can be diffed across
 * engines and commits. Output schema: baremetal.bench/v2 (JSON).
 *
 * NOTE: bm_forward() currently loops bm_step() internally, so "prefill" is not
 * yet batched; this harness deliberately measures the two phases separately so
 * that gap is visible in the numbers.
 */
#include "baremetal.h"
#include "baremetal/context.h"
#include "baremetal/model.h"
#include "baremetal/graph.h"
#include "baremetal/quant.h"
#include "backend/backend.h"
#include "metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

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
        "  --quant q8|q4       quantize matmul weights        (default off)\n"
        "  --avg-w W           average watts (from powermetrics)\n"
        "  --peak-gbps F       peak memory bandwidth for %% of peak (default 0=off)\n"
        "  --peak-gflops F     peak FP32 GFLOPS for %% of peak   (default 0=off)\n"
        "  --power SRC         ac|battery|unknown            (default auto)\n"
        "  --no-raw            omit raw sample arrays         (default on)\n"
        "  --profile           per-kernel GPU timing (counter sampling)\n"
        "  --out FILE          write JSON here (default stdout)\n",
        prog);
}

/* Emit a full stats object: "key": { n, min, max, mean, stddev, p50, p90, p99 } */
static void emit_stats(FILE* f, const char* key, const double* a, int n) {
    fprintf(f, "\"%s\": { \"n\": %d, \"min\": %.3f, \"max\": %.3f, \"mean\": %.3f, "
               "\"stddev\": %.4f, \"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f }",
            key, n, bm_min(a, n), bm_max(a, n), bm_mean(a, n), bm_stddev(a, n),
            bm_percentile_copy(a, n, 50.0),
            bm_percentile_copy(a, n, 90.0),
            bm_percentile_copy(a, n, 99.0));
}

static void emit_arr(FILE* f, const double* a, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, i ? ", %.3f" : "%.3f", a[i]);
    fputc(']', f);
}

/* Total bytes occupied by weight tensors on the GPU, using the same rule the
 * scheduler uses to choose a storage format. */
static size_t model_weight_bytes(bm_model_t* m) {
    bmt_graph_t* g = (bmt_graph_t*)m->graph;
    if (!g) return 0;
    size_t total = 0;
    for (int i = 0; i < g->n_tensors; i++) {
        bmt_tensor_t* t = &g->tensors[i];
        if (t->type != BMT_TENSOR_TYPE_WEIGHT || !t->weight_ptr) continue;
        size_t n = 1;
        for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
        if (n == 0) continue;
        if (t->quantized == 8)                              total += bmt_q8_bytes(n);
        else if (t->quantized == 4)                         total += bmt_q4_bytes(n);
        else if (t->n_dims >= 2 && m->precision != BM_PRECISION_FP32) total += n * 2;
        else                                                total += n * 4;
    }
    return total;
}

typedef struct {
    double utime_ms, stime_ms;
    long minflt, majflt, inblock, oublock, nvcsw, nivcsw;
} rusage_snap_t;

static void snap_rusage(rusage_snap_t* s) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    s->utime_ms = ru.ru_utime.tv_sec * 1000.0 + ru.ru_utime.tv_usec / 1000.0;
    s->stime_ms = ru.ru_stime.tv_sec * 1000.0 + ru.ru_stime.tv_usec / 1000.0;
    s->minflt  = ru.ru_minflt;  s->majflt  = ru.ru_majflt;
    s->inblock = ru.ru_inblock; s->oublock = ru.ru_oublock;
    s->nvcsw   = ru.ru_nvcsw;   s->nivcsw  = ru.ru_nivcsw;
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* out_path  = NULL;
    const char* prec_arg  = NULL;
    const char* power_arg = NULL;
    int   prompt_tokens = 128, gen_tokens = 128, warmup = 2, reps = 5;
    float temp = 0.0f;
    uint64_t seed = 42;
    int   quant = 0;
    int   raw = 1;
    int   profile = 0;    double avg_w = 0.0;
    double peak_gbps = 0.0, peak_gflops = 0.0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)         model_dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt-tokens") && i+1 < argc) prompt_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i+1 < argc)           gen_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc)        warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i+1 < argc)          reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)          temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc)          seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc)     prec_arg = argv[++i];
        else if (!strcmp(argv[i], "--quant") && i+1 < argc) { const char* q = argv[++i];
            quant = (strcmp(q, "q4") == 0) ? 4 : (strcmp(q, "q8") == 0 ? 8 : 0); }
        else if (!strcmp(argv[i], "--avg-w") && i+1 < argc)         avg_w = atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-gbps") && i+1 < argc)     peak_gbps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--peak-gflops") && i+1 < argc)   peak_gflops = atof(argv[++i]);
        else if (!strcmp(argv[i], "--power") && i+1 < argc)         power_arg = argv[++i];
        else if (!strcmp(argv[i], "--no-raw"))                      raw = 0;
        else if (!strcmp(argv[i], "--profile"))                     profile = 1;
        else if (!strcmp(argv[i], "--out") && i+1 < argc)           out_path = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!model_dir) { usage(argv[0]); return 2; }
    if (prompt_tokens < 1) prompt_tokens = 1;
    if (gen_tokens < 2)    gen_tokens = 2;
    if (reps < 1)          reps = 1;

    char started_at[32] = "";
    {
        time_t t = time(NULL);
        strftime(started_at, sizeof(started_at), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    }

    /* ---- model ---- */
    rusage_snap_t rs_start, rs_load, rs_end;
    char therm_start[64] = "";
    snap_rusage(&rs_start);
    bm_probe_line("pmset -g therm", "CPU_Speed_Limit=", therm_start, sizeof(therm_start));

    double t_load0 = bm_now_ms();
    bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
    if (!ctx) { fprintf(stderr, "bm_create failed\n"); return 1; }

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    if (!m) { fprintf(stderr, "model alloc failed\n"); return 1; }
    bm_load_weights(m, model_dir);
    if (!m->weight_buffer) {
        fprintf(stderr, "failed to load weights from %s\n", model_dir);
        return 1;
    }
    m->quant_bits = quant;

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
    snap_rusage(&rs_load);

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
            for (int i = 0; i < prompt_tokens; i++) prompt[i] = (100 + i) % (V > 1 ? V : 2);
        }
        free(ids);
        if (tok) bm_destroy_tokenizer(tok);
    }

    /* ---- measurement buffers ---- */
    int itl_cap = reps * gen_tokens;
    double* prefill_ms = calloc((size_t)reps, sizeof(double));
    double* prefill_tps= calloc((size_t)reps, sizeof(double));
    double* decode_tps = calloc((size_t)reps, sizeof(double));
    double* ttft_ms    = calloc((size_t)reps, sizeof(double));
    double* itl_all    = calloc((size_t)itl_cap, sizeof(double));
    int*    tok_ids    = calloc((size_t)gen_tokens, sizeof(int));
    if (!prefill_ms || !prefill_tps || !decode_tps || !ttft_ms || !itl_all || !tok_ids) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    int itl_n = 0, tok_n = 0;
    double active_s = 0.0, prefill_s_sum = 0.0, decode_s_sum = 0.0;
    long   total_gen_tokens = 0;
    long   active_start_unix = 0, active_end_unix = 0;
    double first_call_ms = 0.0;

    /* ---- warmup (discarded: first passes include pipeline work) ---- */
    for (int w = 0; w < warmup; w++) {
        bm_reset_session(sess);
        double tc0 = bm_now_ms();
        float* lg_pre = bm_forward(sess, prompt, prompt_tokens);
        if (w == 0) first_call_ms = bm_now_ms() - tc0;   /* includes pipeline compile */
        bm_sampler_t* smp = bm_create_sampler(V, temp, 1.0f, seed);
        int tk = bm_sample(smp, lg_pre);
        for (int j = 1; j < gen_tokens; j++) {
            float* lg = bm_step(sess, tk);
            tk = bm_sample(smp, lg);
        }
        bm_destroy_sampler(smp);
    }

    /* ---- optional per-node GPU profile (one decode step) ---- */
    int prof_ok = 0;
    double prof_attn = 0.0, prof_mlp = 0.0, prof_other = 0.0, prof_total = 0.0;
    if (profile) {
        bm_profile_begin(sess);
        bm_reset_session(sess);
        bm_step(sess, prompt[0]);              /* one decode step, profiled */
        bm_profile_end(sess, &prof_attn, &prof_mlp, &prof_other);
        prof_total = prof_attn + prof_mlp + prof_other;
        prof_ok = 1;
    }

    /* ---- measured reps ---- */
    if (ctx->backend_ctx) backend_reset_gpu_timing(ctx->backend_ctx);
    active_start_unix = (long)time(NULL);    for (int r = 0; r < reps; r++) {
        bm_reset_session(sess);
        bm_sampler_t* smp = bm_create_sampler(V, temp, 1.0f, seed);

        double t0 = bm_now_ms();
        float* lg_prefill = bm_forward(sess, prompt, prompt_tokens);
        double t1 = bm_now_ms();
        int tk = bm_sample(smp, lg_prefill);
        double t2 = bm_now_ms();

        prefill_ms[r] = t1 - t0;
        prefill_tps[r] = prefill_ms[r] > 0.0 ? (double)prompt_tokens / (prefill_ms[r] / 1000.0) : 0.0;
        ttft_ms[r]    = t2 - t0;
        if (r == 0 && tok_n < gen_tokens) tok_ids[tok_n++] = tk;

        double dec_ms = 0.0;
        for (int j = 1; j < gen_tokens; j++) {
            double a = bm_now_ms();
            float* lg = bm_step(sess, tk);
            tk = bm_sample(smp, lg);
            double b = bm_now_ms();
            if (itl_n < itl_cap) itl_all[itl_n++] = b - a;
            dec_ms += (b - a);
            if (r == 0 && tok_n < gen_tokens) tok_ids[tok_n++] = tk;
        }
        double t3 = bm_now_ms();

        int decode_steps = gen_tokens - 1;
        decode_tps[r] = dec_ms > 0.0 ? (double)decode_steps / (dec_ms / 1000.0) : 0.0;
        prefill_s_sum += (t1 - t0) / 1000.0;
        decode_s_sum  += dec_ms / 1000.0;
        active_s      += (t3 - t0) / 1000.0;
        total_gen_tokens += gen_tokens;

        bm_destroy_sampler(smp);
    }
    active_end_unix = (long)time(NULL);
    snap_rusage(&rs_end);
    double gpu_busy_ms = ctx->backend_ctx ? backend_get_gpu_busy_ms(ctx->backend_ctx) : 0.0;
    unsigned long long cmd_buffers = ctx->backend_ctx ? backend_get_command_buffers(ctx->backend_ctx) : 0;

    /* ---- memory ---- */
    size_t rss = bm_peak_rss_bytes();
    size_t gpu = ctx->backend_ctx ? backend_get_allocated_memory(ctx->backend_ctx) : 0;
    size_t gpu_peak = ctx->backend_ctx ? backend_get_peak_allocated_memory(ctx->backend_ctx) : 0;
    double kv_bytes_per_token = (double)m->arch.n_layers * 2.0 * (double)m->kv_dim * 4.0;
    double kv_total = kv_bytes_per_token * (double)max_seq;

    /* ---- derived analytics (from measurements already collected) ---- */
    size_t model_bytes = model_weight_bytes(m);
    double bytes_per_param = m->n_parameters ? (double)model_bytes / (double)m->n_parameters : 0.0;
    double bytes_per_tok = (double)model_bytes + kv_bytes_per_token;
    double itl_p50 = bm_percentile_copy(itl_all, itl_n, 50.0);
    double itl_s = itl_p50 / 1000.0;
    double achieved_gbps = itl_s > 0.0 ? bytes_per_tok / itl_s / 1e9 : 0.0;
    double gflops = itl_s > 0.0 ? 2.0 * (double)m->n_parameters / itl_s / 1e9 : 0.0;
    double arith_intensity = bytes_per_tok > 0.0 ? 2.0 * (double)m->n_parameters / bytes_per_tok : 0.0;
    double bw_pct = peak_gbps > 0.0 ? achieved_gbps / peak_gbps * 100.0 : 0.0;
    double mfu_pct = peak_gflops > 0.0 ? gflops / peak_gflops * 100.0 : 0.0;
    int K = itl_n / 2; if (K > 10) K = 10;
    double itl_first10 = 0.0, itl_last10 = 0.0, itl_growth = 0.0;
    if (K > 0) {
        for (int i = 0; i < K; i++) itl_first10 += itl_all[i];
        for (int i = 0; i < K; i++) itl_last10 += itl_all[itl_n - K + i];
        itl_first10 /= K; itl_last10 /= K;
        if (itl_first10 > 0.0) itl_growth = (itl_last10 - itl_first10) / itl_first10 * 100.0;
    }
    double cov_pre  = bm_mean(prefill_ms, reps) != 0.0 ? bm_stddev(prefill_ms, reps) / bm_mean(prefill_ms, reps) * 100.0 : 0.0;
    double cov_dec  = bm_mean(decode_tps, reps)  != 0.0 ? bm_stddev(decode_tps, reps)  / bm_mean(decode_tps, reps)  * 100.0 : 0.0;
    double cov_ttft = bm_mean(ttft_ms, reps)     != 0.0 ? bm_stddev(ttft_ms, reps)     / bm_mean(ttft_ms, reps)     * 100.0 : 0.0;
    double cov_itl  = bm_mean(itl_all, itl_n)    != 0.0 ? bm_stddev(itl_all, itl_n)    / bm_mean(itl_all, itl_n)    * 100.0 : 0.0;

    /* ---- process runtime stats ---- */
    double host_cpu_ms = (rs_end.utime_ms + rs_end.stime_ms) - (rs_start.utime_ms + rs_start.stime_ms);
    double host_cpu_pct = active_s > 0.0 ? host_cpu_ms / (active_s * 1000.0) * 100.0 : 0.0;
    long majflt_load = rs_load.majflt - rs_start.majflt;
    long minflt_load = rs_load.minflt - rs_start.minflt;
    long inblock_load = rs_load.inblock - rs_start.inblock;

    /* ---- GPU execution vs host time ---- */
    double active_ms = active_s * 1000.0;
    double host_ms = active_ms > gpu_busy_ms ? active_ms - gpu_busy_ms : 0.0;
    double busy_pct = active_ms > 0.0 ? gpu_busy_ms / active_ms * 100.0 : 0.0;
    double buffers_per_tok = total_gen_tokens > 0 ? (double)cmd_buffers / (double)total_gen_tokens : 0.0;

    /* ---- energy (from powermetrics watts supplied by the wrapper) ---- */
    double energy_joules = 0.0, j_per_token = 0.0;
    if (avg_w > 0.0 && active_s > 0.0) {
        energy_joules = avg_w * active_s;
        if (total_gen_tokens > 0) j_per_token = energy_joules / (double)total_gen_tokens;
    }

    /* ---- host metadata ---- */
    char cpu[128] = "", osver[64] = "", git[64] = "";
    char gpu_name[128] = "", gpu_cores_s[32] = "", batt[128] = "", therm[64] = "";
    bm_sysctl_str("machdep.cpu.brand_string", cpu, sizeof(cpu));
    bm_sysctl_str("kern.osproductversion", osver, sizeof(osver));
    unsigned long long n_cpu    = bm_sysctl_u64("hw.ncpu");
    unsigned long long ram      = bm_sysctl_u64("hw.memsize");
    unsigned long long p_cores = bm_sysctl_u64("hw.perflevel0.logicalcpu");
    unsigned long long e_cores = bm_sysctl_u64("hw.perflevel1.logicalcpu");
    bm_probe_line("system_profiler SPDisplaysDataType 2>/dev/null",
                  "Chipset Model:", gpu_name, sizeof(gpu_name));
    bm_probe_line("system_profiler SPDisplaysDataType 2>/dev/null",
                  "Total Number of Cores:", gpu_cores_s, sizeof(gpu_cores_s));
    int gpu_cores = atoi(gpu_cores_s);
    if (!power_arg) {
        bm_probe_line("pmset -g batt", "Now drawing from", batt, sizeof(batt));
        if (strstr(batt, "AC Power")) power_arg = "ac";
        else if (strstr(batt, "Battery Power")) power_arg = "battery";
        else power_arg = "unknown";
    }
    bm_probe_line("pmset -g therm", "CPU_Speed_Limit=", therm, sizeof(therm));

    /* ---- system context ---- */
    double load[3] = { 0, 0, 0 };
    getloadavg(load, 3);
    char load_avg[64];
    snprintf(load_avg, sizeof(load_avg), "%.2f %.2f %.2f", load[0], load[1], load[2]);

    unsigned long long swap_total = 0, swap_used = 0;
    {
        struct xsw_usage xsw; size_t len = sizeof(xsw);
        if (sysctlbyname("vm.swapusage", &xsw, &len, NULL, 0) == 0) {
            swap_total = xsw.xsu_total; swap_used = xsw.xsu_used;
        }
    }

    unsigned long long pagesize = bm_sysctl_u64("hw.pagesize");
    unsigned long long cacheline = bm_sysctl_u64("hw.cachelinesize");
    char hw_model[64] = "", hw_machine[32] = "", os_build[64] = "";
    bm_sysctl_str("hw.model", hw_model, sizeof(hw_model));
    bm_sysctl_str("hw.machine", hw_machine, sizeof(hw_machine));
    bm_sysctl_str("kern.osversion", os_build, sizeof(os_build));
    unsigned long long free_bytes = 0, compressed_bytes = 0;
    {
        char vmp[64];
        bm_probe_line("vm_stat", "Pages free:", vmp, sizeof(vmp));
        free_bytes = strtoull(vmp, NULL, 10) * pagesize;
        bm_probe_line("vm_stat", "Pages occupied by compressor:", vmp, sizeof(vmp));
        compressed_bytes = strtoull(vmp, NULL, 10) * pagesize;
    }

    long uptime_s = 0;
    {
        struct timeval bt; size_t len = sizeof(bt);
        if (sysctlbyname("kern.boottime", &bt, &len, NULL, 0) == 0)
            uptime_s = (long)time(NULL) - (long)bt.tv_sec;
    }

    int battery_pct = -1;
    {
        FILE* p = popen("pmset -g batt", "r");
        if (p) {
            char line[256];
            while (fgets(line, sizeof(line), p)) {
                char* pct = strchr(line, '%');
                if (pct) {
                    char* s = pct;
                    while (s > line && s[-1] >= '0' && s[-1] <= '9') s--;
                    if (s < pct) battery_pct = atoi(s);
                    break;
                }
            }
            pclose(p);
        }
    }

    char clang_ver[128] = "", metal_ver[128] = "";
    bm_probe_line("clang --version 2>/dev/null", "version", clang_ver, sizeof(clang_ver));
    bm_probe_line("xcrun -sdk macosx metal --version 2>/dev/null", "version", metal_ver, sizeof(metal_ver));

    backend_device_info_t dinfo;
    backend_get_device_info(ctx->backend_ctx, &dinfo);
    {
        FILE* p = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (p) { if (fgets(git, sizeof(git), p)) { char* nl = strchr(git, '\n'); if (nl) *nl = 0; } pclose(p); }
    }
    char git_branch[64] = "";
    {
        FILE* p = popen("git rev-parse --abbrev-ref HEAD 2>/dev/null", "r");
        if (p) { if (fgets(git_branch, sizeof(git_branch), p)) { char* nl = strchr(git_branch, '\n'); if (nl) *nl = 0; } pclose(p); }
    }
    int git_dirty = 0;
    {
        FILE* p = popen("git status --porcelain 2>/dev/null", "r");
        if (p) { char l[8]; git_dirty = fgets(l, sizeof(l), p) ? 1 : 0; pclose(p); }
    }
    char ended_at[32] = "";
    {
        time_t t = time(NULL);
        strftime(ended_at, sizeof(ended_at), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    }
    char cpu_e[256], osver_e[128], git_e[128], gpu_e[256], therm_e[128], model_e[512];
    char clang_e[256], metal_e[256];
    char hwmodel_e[128], machine_e[64], osbuild_e[128], branch_e[128], thermstart_e[128];
    bm_json_escape(cpu, cpu_e, sizeof(cpu_e));
    bm_json_escape(osver, osver_e, sizeof(osver_e));
    bm_json_escape(git, git_e, sizeof(git_e));
    bm_json_escape(gpu_name, gpu_e, sizeof(gpu_e));
    bm_json_escape(therm, therm_e, sizeof(therm_e));
    bm_json_escape(model_dir, model_e, sizeof(model_e));
    bm_json_escape(clang_ver, clang_e, sizeof(clang_e));
    bm_json_escape(metal_ver, metal_e, sizeof(metal_e));
    bm_json_escape(hw_model, hwmodel_e, sizeof(hwmodel_e));
    bm_json_escape(hw_machine, machine_e, sizeof(machine_e));
    bm_json_escape(os_build, osbuild_e, sizeof(osbuild_e));
    bm_json_escape(git_branch, branch_e, sizeof(branch_e));
    bm_json_escape(therm_start, thermstart_e, sizeof(thermstart_e));

    /* ---- emit JSON (baremetal.bench/v2) ---- */
    FILE* out = out_path ? fopen(out_path, "w") : stdout;
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema\": \"baremetal.bench/v2\",\n");
    fprintf(out, "  \"meta\": {\n");
    fprintf(out, "    \"engine\": \"bare.metal\",\n");
    fprintf(out, "    \"git_commit\": \"%s\",\n", git_e);
    fprintf(out, "    \"git_branch\": \"%s\",\n", branch_e);
    fprintf(out, "    \"git_dirty\": %d,\n", git_dirty);
    fprintf(out, "    \"host\": \"%s\",\n", cpu_e);
    fprintf(out, "    \"hw_model\": \"%s\",\n", hwmodel_e);
    fprintf(out, "    \"hw_machine\": \"%s\",\n", machine_e);
    fprintf(out, "    \"os_build\": \"%s\",\n", osbuild_e);
    fprintf(out, "    \"pagesize\": %llu,\n", pagesize);
    fprintf(out, "    \"cacheline\": %llu,\n", cacheline);
    fprintf(out, "    \"cpu_count\": %llu,\n", n_cpu);
    fprintf(out, "    \"p_cores\": %llu,\n", p_cores);
    fprintf(out, "    \"e_cores\": %llu,\n", e_cores);
    fprintf(out, "    \"gpu_name\": \"%s\",\n", gpu_e);
    fprintf(out, "    \"gpu_cores\": %d,\n", gpu_cores);
    fprintf(out, "    \"gpu_max_threads_per_tg\": %d,\n", dinfo.max_threads_per_threadgroup);
    fprintf(out, "    \"gpu_max_threadgroup_mem\": %zu,\n", dinfo.max_threadgroup_memory);
    fprintf(out, "    \"gpu_max_buffer_bytes\": %zu,\n", dinfo.max_buffer_bytes);
    fprintf(out, "    \"gpu_recommended_working_set\": %zu,\n", dinfo.recommended_max_working_set);
    fprintf(out, "    \"gpu_has_unified_memory\": %d,\n", dinfo.has_unified_memory);
    fprintf(out, "    \"ram_bytes\": %llu,\n", ram);
    fprintf(out, "    \"macos\": \"%s\",\n", osver_e);
    fprintf(out, "    \"power\": \"%s\",\n", power_arg);
    fprintf(out, "    \"thermal_speed_limit\": \"%s\",\n", therm_e);
    fprintf(out, "    \"thermal_start\": \"%s\",\n", thermstart_e);
    fprintf(out, "    \"load_avg\": \"%s\",\n", load_avg);
    fprintf(out, "    \"uptime_seconds\": %ld,\n", uptime_s);
    fprintf(out, "    \"battery_pct\": %d,\n", battery_pct);
    fprintf(out, "    \"swap_total_bytes\": %llu,\n", swap_total);
    fprintf(out, "    \"swap_used_bytes\": %llu,\n", swap_used);
    fprintf(out, "    \"free_bytes\": %llu,\n", free_bytes);
    fprintf(out, "    \"compressed_bytes\": %llu,\n", compressed_bytes);
    fprintf(out, "    \"clang_version\": \"%s\",\n", clang_e);
    fprintf(out, "    \"metal_version\": \"%s\",\n", metal_e);
    fprintf(out, "    \"started_at\": \"%s\",\n", started_at);
    fprintf(out, "    \"ended_at\": \"%s\",\n", ended_at);
    fprintf(out, "    \"model\": \"%s\",\n", model_e);
    fprintf(out, "    \"params\": %zu,\n", m->n_parameters);
    fprintf(out, "    \"n_layers\": %d,\n", m->arch.n_layers);
    fprintf(out, "    \"dim\": %d,\n", m->arch.dim);
    fprintf(out, "    \"n_heads\": %d,\n", m->arch.n_heads);
    fprintf(out, "    \"n_kv_heads\": %d,\n", m->arch.n_kv_heads);
    fprintf(out, "    \"max_seq\": %d,\n", max_seq);
    fprintf(out, "    \"precision\": \"%s\",\n", precision_name(m->precision));
    fprintf(out, "    \"quant\": \"%s\"\n", quant == 8 ? "q8" : quant == 4 ? "q4" : "none");
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
    fprintf(out, "    \"load_ms\": %.3f,\n", load_ms);
    fprintf(out, "    \"prefill\": {\n      ");
    emit_stats(out, "tok_s", prefill_tps, reps);
    fprintf(out, ",\n      ");
    emit_stats(out, "ms", prefill_ms, reps);
    fprintf(out, "\n    },\n");
    fprintf(out, "    \"decode\": {\n      ");
    emit_stats(out, "tok_s", decode_tps, reps);
    fprintf(out, "\n    },\n    ");
    emit_stats(out, "ttft_ms", ttft_ms, reps);
    fprintf(out, ",\n    ");
    emit_stats(out, "itl_ms", itl_all, itl_n);
    fprintf(out, ",\n");
    if (raw) {
        fprintf(out, "    \"samples\": {\n");
        fprintf(out, "      \"prefill_ms\": ");  emit_arr(out, prefill_ms, reps);  fprintf(out, ",\n");
        fprintf(out, "      \"prefill_tps\": ");  emit_arr(out, prefill_tps, reps); fprintf(out, ",\n");
        fprintf(out, "      \"decode_tps\": ");  emit_arr(out, decode_tps, reps);  fprintf(out, ",\n");
        fprintf(out, "      \"ttft_ms\": ");     emit_arr(out, ttft_ms, reps);     fprintf(out, ",\n");
        fprintf(out, "      \"itl_ms\": ");      emit_arr(out, itl_all, itl_n);    fprintf(out, "\n");
        fprintf(out, "    },\n");
    }
    fprintf(out, "    \"generated_token_ids\": [");
    for (int i = 0; i < tok_n; i++) fprintf(out, i ? ", %d" : "%d", tok_ids[i]);
    fprintf(out, "],\n");
    fprintf(out, "    \"memory\": {\n");
    fprintf(out, "      \"rss_peak_bytes\": %zu,\n", rss);
    fprintf(out, "      \"gpu_alloc_bytes\": %zu,\n", gpu);
    fprintf(out, "      \"gpu_peak_bytes\": %zu,\n", gpu_peak);
    fprintf(out, "      \"kv_bytes_per_token\": %.0f,\n", kv_bytes_per_token);
    fprintf(out, "      \"kv_total_bytes\": %.0f\n", kv_total);
    fprintf(out, "    },\n");
    fprintf(out, "    \"analysis\": {\n");
    fprintf(out, "      \"model_bytes\": %zu,\n", model_bytes);
    fprintf(out, "      \"bytes_per_param\": %.3f,\n", bytes_per_param);
    fprintf(out, "      \"bytes_per_decode_token\": %.0f,\n", bytes_per_tok);
    fprintf(out, "      \"achieved_gbps\": %.2f,\n", achieved_gbps);
    fprintf(out, "      \"gflops\": %.2f,\n", gflops);
    fprintf(out, "      \"arith_intensity\": %.2f,\n", arith_intensity);
    fprintf(out, "      \"bw_pct_of_peak\": %.1f,\n", bw_pct);
    fprintf(out, "      \"mfu_pct_of_peak\": %.1f,\n", mfu_pct);
    fprintf(out, "      \"itl_first10_ms\": %.3f,\n", itl_first10);
    fprintf(out, "      \"itl_last10_ms\": %.3f,\n", itl_last10);
    fprintf(out, "      \"itl_growth_pct\": %.2f,\n", itl_growth);
    fprintf(out, "      \"cov\": { \"prefill\": %.2f, \"decode\": %.2f, \"ttft\": %.2f, \"itl\": %.2f }\n",
            cov_pre, cov_dec, cov_ttft, cov_itl);
    fprintf(out, "    },\n");
    fprintf(out, "    \"gpu_exec\": {\n");
    fprintf(out, "      \"busy_ms\": %.3f,\n", gpu_busy_ms);
    fprintf(out, "      \"host_ms\": %.3f,\n", host_ms);
    fprintf(out, "      \"busy_pct\": %.2f,\n", busy_pct);
    fprintf(out, "      \"command_buffers\": %llu,\n", cmd_buffers);
    fprintf(out, "      \"buffers_per_decode_token\": %.2f\n", buffers_per_tok);
    fprintf(out, "    },\n");
    fprintf(out, "    \"energy\": { \"avg_w\": %.2f, \"joules\": %.2f, \"j_per_token\": %.4f },\n",
            avg_w, energy_joules, j_per_token);
    fprintf(out, "    \"active_seconds\": %.3f,\n", active_s);
    fprintf(out, "    \"active_started_unix\": %ld,\n", active_start_unix);
    fprintf(out, "    \"active_ended_unix\": %ld,\n", active_end_unix);
    fprintf(out, "    \"prefill_seconds\": %.3f,\n", prefill_s_sum);
    fprintf(out, "    \"decode_seconds\": %.3f,\n", decode_s_sum);
    fprintf(out, "    \"first_call_ms\": %.3f,\n", first_call_ms);
    fprintf(out, "    \"process\": {\n");
    fprintf(out, "      \"user_ms\": %.3f,\n", rs_end.utime_ms - rs_start.utime_ms);
    fprintf(out, "      \"sys_ms\": %.3f,\n", rs_end.stime_ms - rs_start.stime_ms);
    fprintf(out, "      \"host_cpu_ms\": %.3f,\n", host_cpu_ms);
    fprintf(out, "      \"host_cpu_pct\": %.2f,\n", host_cpu_pct);
    fprintf(out, "      \"minflt\": %ld,\n", rs_end.minflt - rs_start.minflt);
    fprintf(out, "      \"majflt\": %ld,\n", rs_end.majflt - rs_start.majflt);
    fprintf(out, "      \"inblock\": %ld,\n", rs_end.inblock - rs_start.inblock);
    fprintf(out, "      \"oublock\": %ld,\n", rs_end.oublock - rs_start.oublock);
    fprintf(out, "      \"nvcsw\": %ld,\n", rs_end.nvcsw - rs_start.nvcsw);
    fprintf(out, "      \"nivcsw\": %ld,\n", rs_end.nivcsw - rs_start.nivcsw);
    fprintf(out, "      \"minflt_load\": %ld,\n", minflt_load);
    fprintf(out, "      \"majflt_load\": %ld,\n", majflt_load);
    fprintf(out, "      \"inblock_load\": %ld\n", inblock_load);
    fprintf(out, "    },\n");
    if (prof_ok) {
        double attn_pct = prof_total > 0.0 ? prof_attn / prof_total * 100.0 : 0.0;
        double mlp_pct  = prof_total > 0.0 ? prof_mlp  / prof_total * 100.0 : 0.0;
        fprintf(out, "    \"kernel_profile\": {\n");
        fprintf(out, "      \"attention_ms\": %.4f,\n", prof_attn);
        fprintf(out, "      \"mlp_ms\": %.4f,\n", prof_mlp);
        fprintf(out, "      \"other_ms\": %.4f,\n", prof_other);
        fprintf(out, "      \"total_ms\": %.4f,\n", prof_total);
        fprintf(out, "      \"attention_pct\": %.1f,\n", attn_pct);
        fprintf(out, "      \"mlp_pct\": %.1f\n", mlp_pct);
        fprintf(out, "    },\n");
    }
    fprintf(out, "    \"generated_tokens\": %ld\n", total_gen_tokens);
    fprintf(out, "  }\n");
    fprintf(out, "}\n");
    if (out != stdout) fclose(out);

    free(prefill_ms); free(prefill_tps); free(decode_tps); free(ttft_ms);
    free(itl_all); free(tok_ids); free(prompt);
    bm_destroy_session(sess);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return 0;
}
