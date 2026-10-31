/* bench/train_bench.c -- bare.metal training benchmark.
 *
 * Measures optimizer step time, tokens/s, loss, memory and GPU occupancy for a
 * fixed token stream. Emits baremetal.bench/v2 JSON with the training metrics
 * under results.train so the existing run/aggregate/publish pipeline can carry
 * them alongside inference results.
 */
#include "baremetal.h"
#include "baremetal/context.h"
#include "baremetal/model.h"
#include "baremetal/tokenizer.h"
#include "baremetal/json.h"
#include "baremetal/trainer.h"
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
        "  --data FILE        training text (default test/grad/g4_text.txt)\n"
        "  --seq-len N        sequence length               (default 64)\n"
        "  --steps N          measured optimizer steps      (default 20)\n"
        "  --warmup N         discarded warmup steps        (default 2)\n"
        "  --batch-size N     sequences per optimizer step  (default 1)\n"
        "  --lr F             learning rate                 (default 1e-4)\n"
        "  --precision P      fp32|fp16|bf16 (default device)\n"
        "  --avg-w W          average watts (from powermetrics)\n"
        "  --ref FILE         prior JSON for loss parity\n"
        "  --parity-steps N   losses compared to ref      (default 1)\n"
        "  --tol F            parity relative tolerance    (default 1e-6)\n"
        "  --check            exit non-zero if parity fails\n"
        "  --no-raw           omit per-step loss array\n"
        "  --out FILE         write JSON here (default stdout)\n",
        prog);
}

static void emit_stats(FILE* f, const char* key, const double* a, int n) {
    fprintf(f, "\"%s\": { \"n\": %d, \"min\": %.3f, \"max\": %.3f, \"mean\": %.3f, "
               "\"stddev\": %.4f, \"p50\": %.3f, \"p90\": %.3f, \"p99\": %.3f }",
            key, n, bm_min(a, n), bm_max(a, n), bm_mean(a, n), bm_stddev(a, n),
            bm_percentile_copy(a, n, 50.0),
            bm_percentile_copy(a, n, 90.0),
            bm_percentile_copy(a, n, 99.0));
}

static void emit_arr(FILE* f, const float* a, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, i ? ", %.9g" : "%.9g", (double)a[i]);
    fputc(']', f);
}

static void json_git(const char* args, char* out, size_t n) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "git %s 2>/dev/null", args);
    bm_probe_line(cmd, "", out, n);
}

static float* load_ref_losses(const char* path, int* count) {
    *count = 0;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char* text = malloc((size_t)sz + 1);
    if (!text) { fclose(f); return NULL; }
    if (fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        free(text); fclose(f); return NULL;
    }
    text[sz] = '\0';
    fclose(f);

    char err[128];
    bmt_json_t* root = bmt_json_parse(text, err, sizeof(err));
    free(text);
    if (!root) return NULL;
    const bmt_json_t* train = bmt_json_get(root, "results");
    train = train ? bmt_json_get(train, "train") : NULL;
    const bmt_json_t* loss = train ? bmt_json_get(train, "loss") : NULL;
    const bmt_json_t* vals = loss ? bmt_json_get(loss, "values") : NULL;
    int n = vals ? bmt_json_array_count(vals) : 0;
    float* out = n > 0 ? malloc((size_t)n * sizeof(float)) : NULL;
    for (int i = 0; out && i < n; i++) {
        float v = 0.0f;
        if (bmt_json_float(bmt_json_array_get(vals, i), &v) == 0) out[i] = v;
    }
    bmt_json_free(root);
    *count = n;
    return out;
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* data_path = "test/grad/g4_text.txt";
    const char* out_path = NULL;
    const char* ref_path = NULL;
    const char* prec_arg = NULL;
    int seq_len = 64, steps = 20, warmup = 2, batch_size = 1;
    int raw = 1, check = 0, parity_steps = 1;
    float lr = 1e-4f, tol = 1e-6f;
    double avg_w = 0.0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model") && i+1 < argc)      model_dir = argv[++i];
        else if (!strcmp(argv[i], "--data") && i+1 < argc)       data_path = argv[++i];
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc)    seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc)      steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc)     warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-size") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc)         lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc)  prec_arg = argv[++i];
        else if (!strcmp(argv[i], "--avg-w") && i+1 < argc)      avg_w = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ref") && i+1 < argc)        ref_path = argv[++i];
        else if (!strcmp(argv[i], "--parity-steps") && i+1 < argc) parity_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && i+1 < argc)         tol = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--check"))                     check = 1;
        else if (!strcmp(argv[i], "--no-raw"))                    raw = 0;
        else if (!strcmp(argv[i], "--out") && i+1 < argc)        out_path = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    if (!model_dir) { usage(argv[0]); return 2; }
    if (seq_len < 1) seq_len = 1;
    if (steps < 1) steps = 1;
    if (warmup < 0) warmup = 0;
    if (batch_size < 1) batch_size = 1;
    if (parity_steps < 1) parity_steps = 1;
    if (tol <= 0.0f) tol = 1e-6f;

    char started_at[32] = "";
    {
        time_t t = time(NULL);
        strftime(started_at, sizeof(started_at), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    }

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
    if (prec_arg) {
        if      (!strcmp(prec_arg, "bf16")) m->precision = BM_PRECISION_BF16;
        else if (!strcmp(prec_arg, "fp16")) m->precision = BM_PRECISION_FP16;
        else                                m->precision = BM_PRECISION_FP32;
    } else {
        m->precision = bm_get_supported_precision(ctx);
    }

    bm_train_config_t cfg = {0};
    cfg.learning_rate = lr;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.epsilon = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.grad_clip = 1.0f;
    cfg.warmup_steps = 0;
    cfg.max_steps = steps;
    cfg.grad_accum_steps = 1;
    cfg.seq_len = seq_len;

    bm_trainer_t* trainer = bm_create_trainer(ctx, m, &cfg);
    if (!trainer) { fprintf(stderr, "trainer creation failed\n"); return 1; }
    double load_ms = bm_now_ms() - t_load0;

    int V = m->arch.vocab_size;
    int n_tok = 0;
    int* tokens = NULL;
    bm_tokenizer_t* tok = bm_create_tokenizer(model_dir);
    if (tok) {
        FILE* f = fopen(data_path, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz > 0) {
                char* text = malloc((size_t)sz + 1);
                if (text && fread(text, 1, (size_t)sz, f) == (size_t)sz) {
                    text[sz] = '\0';
                    tokens = bm_encode(tok, text, &n_tok);
                }
                free(text);
            }
            fclose(f);
        }
        bm_destroy_tokenizer(tok);
    }
    if (!tokens || n_tok < 2) {
        free(tokens);
        n_tok = 4096;
        tokens = malloc((size_t)n_tok * sizeof(int));
        uint32_t s = 42;
        for (int i = 0; tokens && i < n_tok; i++) {
            s = s * 1664525u + 1013904223u;
            tokens[i] = (int)(s % (uint32_t)V);
        }
    }
    if (!tokens) { fprintf(stderr, "token stream failed\n"); return 1; }

    int* inputs = malloc((size_t)batch_size * seq_len * sizeof(int));
    int* targets = malloc((size_t)batch_size * seq_len * sizeof(int));
    double* step_ms = calloc((size_t)steps, sizeof(double));
    double* token_rate = calloc((size_t)steps, sizeof(double));
    float* losses = calloc((size_t)steps, sizeof(float));
    int* cp_step = calloc((size_t)steps, sizeof(int));
    float* cp_loss = calloc((size_t)steps, sizeof(float));
    if (!inputs || !targets || !step_ms || !token_rate || !losses || !cp_step || !cp_loss) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    int pos = 0;
    int total_tokens = 0;
    time_t active_start_unix = 0, active_end_unix = 0;
    for (int step = -warmup; step < steps; step++) {
        if (step == 0) active_start_unix = time(NULL);
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                inputs[b * seq_len + i] = tokens[pos % n_tok];
                targets[b * seq_len + i] = tokens[(pos + 1) % n_tok];
                pos++;
            }
        }
        double t0 = bm_now_ms();
        float loss = bm_train_step(trainer, inputs, targets, batch_size, seq_len);
        double ms = bm_now_ms() - t0;
        if (step >= 0) {
            step_ms[step] = ms;
            token_rate[step] = (double)(batch_size * seq_len) * 1000.0 / (ms > 0.0 ? ms : 1.0);
            losses[step] = loss;
            total_tokens += batch_size * seq_len;
        }
    }
    active_end_unix = time(NULL);

    int n_cp = 0;
    int marks[5] = { 0, steps / 4, steps / 2, (3 * steps) / 4, steps - 1 };
    for (int i = 0; i < 5; i++) {
        int s = marks[i];
        if (s < 0 || s >= steps) continue;
        int seen = 0;
        for (int j = 0; j < n_cp; j++) if (cp_step[j] == s) seen = 1;
        if (seen) continue;
        cp_step[n_cp] = s;
        cp_loss[n_cp] = losses[s];
        n_cp++;
    }

    float loss_min = losses[0], loss_mean = 0.0f;
    for (int i = 0; i < steps; i++) {
        if (losses[i] < loss_min) loss_min = losses[i];
        loss_mean += losses[i];
    }
    loss_mean /= (float)steps;

    int ref_n = 0;
    float* ref_losses = ref_path ? load_ref_losses(ref_path, &ref_n) : NULL;
    int drift_n = ref_losses ? (ref_n < steps ? ref_n : steps) : 0;
    int parity_n = drift_n < parity_steps ? drift_n : parity_steps;
    double parity_max_rel = 0.0, drift_max_rel = 0.0;
    int parity_pass = parity_n > 0;
    for (int i = 0; i < drift_n; i++) {
        double denom = fabs((double)ref_losses[i]);
        if (denom < 1e-6) denom = 1e-6;
        double rel = fabs((double)losses[i] - (double)ref_losses[i]) / denom;
        if (rel > drift_max_rel) drift_max_rel = rel;
        if (i < parity_n) {
            if (rel > parity_max_rel) parity_max_rel = rel;
            if (rel > (double)tol) parity_pass = 0;
        }
    }
    if (ref_path && parity_n == 0) parity_pass = 0;

    double step_sum = 0.0;
    for (int i = 0; i < steps; i++) step_sum += step_ms[i];
    double active_s = step_sum / 1000.0;
    double gpu_ms = backend_get_gpu_busy_ms(ctx->backend_ctx);
    uint64_t cmds = backend_get_command_buffers(ctx->backend_ctx);
    double busy_pct = active_s > 0.0 ? (gpu_ms / (active_s * 1000.0)) * 100.0 : 0.0;
    if (busy_pct > 100.0) busy_pct = 100.0;

    size_t rss_peak = bm_peak_rss_bytes();
    size_t gpu_peak = backend_get_peak_allocated_memory(ctx->backend_ctx);
    size_t gpu_alloc = backend_get_allocated_memory(ctx->backend_ctx);

    char cpu[128] = "", osver[64] = "", gpu_name[128] = "";
    char hw_model[64] = "", hw_machine[32] = "", os_build[64] = "";
    bm_sysctl_str("machdep.cpu.brand_string", cpu, sizeof(cpu));
    bm_sysctl_str("hw.model", hw_model, sizeof(hw_model));
    bm_sysctl_str("hw.machine", hw_machine, sizeof(hw_machine));
    bm_sysctl_str("kern.osproductversion", osver, sizeof(osver));
    bm_sysctl_str("kern.osversion", os_build, sizeof(os_build));
    bm_probe_line("system_profiler SPDisplaysDataType 2>/dev/null",
                  "Chipset Model:", gpu_name, sizeof(gpu_name));
    unsigned long long n_cpu = bm_sysctl_u64("hw.ncpu");
    unsigned long long ram = bm_sysctl_u64("hw.memsize");
    unsigned long long p_cores = bm_sysctl_u64("hw.perflevel0.logicalcpu");
    unsigned long long e_cores = bm_sysctl_u64("hw.perflevel1.logicalcpu");

    char cpu_e[256], osver_e[128], gpu_e[256], hw_e[128], machine_e[64], build_e[128], model_e[512];
    bm_json_escape(cpu, cpu_e, sizeof(cpu_e));
    bm_json_escape(osver, osver_e, sizeof(osver_e));
    bm_json_escape(gpu_name, gpu_e, sizeof(gpu_e));
    bm_json_escape(hw_model, hw_e, sizeof(hw_e));
    bm_json_escape(hw_machine, machine_e, sizeof(machine_e));
    bm_json_escape(os_build, build_e, sizeof(build_e));
    bm_json_escape(model_dir, model_e, sizeof(model_e));

    char git_commit[64] = "", git_branch[64] = "";
    json_git("rev-parse --short HEAD", git_commit, sizeof(git_commit));
    json_git("rev-parse --abbrev-ref HEAD", git_branch, sizeof(git_branch));
    int git_dirty = 0;
    {
        FILE* p = popen("git status --porcelain 2>/dev/null", "r");
        if (p) { char l[8]; git_dirty = fgets(l, sizeof(l), p) ? 1 : 0; pclose(p); }
    }
    char commit_e[128], branch_e[128];
    bm_json_escape(git_commit, commit_e, sizeof(commit_e));
    bm_json_escape(git_branch, branch_e, sizeof(branch_e));

    char ended_at[32] = "";
    {
        time_t t = time(NULL);
        strftime(ended_at, sizeof(ended_at), "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    }

    FILE* out = out_path ? fopen(out_path, "w") : stdout;
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema\": \"baremetal.bench/v2\",\n");
    fprintf(out, "  \"meta\": {\n");
    fprintf(out, "    \"engine\": \"bare.metal\",\n");
    fprintf(out, "    \"git_commit\": \"%s\",\n", commit_e);
    fprintf(out, "    \"git_branch\": \"%s\",\n", branch_e);
    fprintf(out, "    \"git_dirty\": %d,\n", git_dirty);
    fprintf(out, "    \"host\": \"%s\",\n", cpu_e);
    fprintf(out, "    \"hw_model\": \"%s\",\n", hw_e);
    fprintf(out, "    \"hw_machine\": \"%s\",\n", machine_e);
    fprintf(out, "    \"os_build\": \"%s\",\n", build_e);
    fprintf(out, "    \"gpu_name\": \"%s\",\n", gpu_e);
    fprintf(out, "    \"ram_bytes\": %llu,\n", ram);
    fprintf(out, "    \"cpu_count\": %llu,\n", n_cpu);
    fprintf(out, "    \"p_cores\": %llu,\n", p_cores);
    fprintf(out, "    \"e_cores\": %llu,\n", e_cores);
    fprintf(out, "    \"macos\": \"%s\",\n", osver_e);
    fprintf(out, "    \"started_at\": \"%s\",\n", started_at);
    fprintf(out, "    \"ended_at\": \"%s\",\n", ended_at);
    fprintf(out, "    \"model\": \"%s\",\n", model_e);
    fprintf(out, "    \"params\": %zu,\n", m->n_parameters);
    fprintf(out, "    \"n_layers\": %d,\n", m->arch.n_layers);
    fprintf(out, "    \"dim\": %d,\n", m->arch.dim);
    fprintf(out, "    \"n_heads\": %d,\n", m->arch.n_heads);
    fprintf(out, "    \"n_kv_heads\": %d,\n", m->arch.n_kv_heads);
    fprintf(out, "    \"max_seq\": %d,\n", m->arch.max_seq_len);
    fprintf(out, "    \"precision\": \"%s\",\n", precision_name(m->precision));
    fprintf(out, "    \"quant\": \"none\"\n");
    fprintf(out, "  },\n");

    fprintf(out, "  \"config\": {\n");
    fprintf(out, "    \"stage\": \"train\",\n");
    fprintf(out, "    \"seq_len\": %d,\n", seq_len);
    fprintf(out, "    \"batch_size\": %d,\n", batch_size);
    fprintf(out, "    \"steps\": %d,\n", steps);
    fprintf(out, "    \"warmup\": %d,\n", warmup);
    fprintf(out, "    \"learning_rate\": %.9g,\n", (double)lr);
    fprintf(out, "    \"seed\": 42,\n");
    fprintf(out, "    \"data\": \"%s\"\n", data_path);
    fprintf(out, "  },\n");

    fprintf(out, "  \"results\": {\n");
    fprintf(out, "    \"train\": {\n");
    fprintf(out, "      ");
    emit_stats(out, "step_ms", step_ms, steps);
    fprintf(out, ",\n      ");
    emit_stats(out, "tokens_per_s", token_rate, steps);
    fprintf(out, ",\n      \"loss\": { \"first\": %.9g, \"last\": %.9g, \"min\": %.9g, \"mean\": %.9g",
            (double)losses[0], (double)losses[steps-1], (double)loss_min, (double)loss_mean);
    if (raw) {
        fprintf(out, ", \"values\": ");
        emit_arr(out, losses, steps);
    }
    fprintf(out, " },\n      \"checkpoints\": [");
    for (int i = 0; i < n_cp; i++)
        fprintf(out, "%s{\"step\": %d, \"loss\": %.9g}", i ? ", " : "", cp_step[i], (double)cp_loss[i]);
    fprintf(out, "],\n      \"total_tokens\": %d,\n", total_tokens);
    if (ref_path) {
        const char* ref_name = strrchr(ref_path, '/');
        ref_name = ref_name ? ref_name + 1 : ref_path;
        fprintf(out, "      \"parity\": { \"ref\": \"%s\", \"n\": %d, \"tol\": %.9g, "
                     "\"max_rel_err\": %.9g, \"pass\": %s, \"drift_n\": %d, \"drift_max_rel_err\": %.9g },\n",
                ref_name, parity_n, (double)tol, parity_max_rel, parity_pass ? "true" : "false",
                drift_n, drift_max_rel);
    }
    fprintf(out, "      \"load_ms\": %.3f\n", load_ms);
    fprintf(out, "    },\n");
    fprintf(out, "    \"memory\": { \"rss_peak_bytes\": %zu, \"gpu_peak_bytes\": %zu, \"gpu_alloc_bytes\": %zu },\n",
            rss_peak, gpu_peak, gpu_alloc);
    fprintf(out, "    \"gpu_exec\": { \"busy_ms\": %.3f, \"busy_pct\": %.2f, \"cmd_buffers\": %llu, "
                 "\"buffers_per_step\": %.3f },\n",
            gpu_ms, busy_pct, (unsigned long long)cmds,
            steps > 0 ? (double)cmds / (double)steps : 0.0);
    fprintf(out, "    \"active_seconds\": %.6f,\n", active_s);
    fprintf(out, "    \"active_started_unix\": %ld,\n", (long)active_start_unix);
    fprintf(out, "    \"active_ended_unix\": %ld,\n", (long)active_end_unix);
    if (avg_w > 0.0) {
        double joules = avg_w * active_s;
        double jpt = total_tokens > 0 ? joules / (double)total_tokens : 0.0;
        fprintf(out, "    \"energy\": { \"avg_w\": %.3f, \"joules\": %.3f, \"j_per_token\": %.6f, "
                     "\"j_per_step\": %.6f },\n", avg_w, joules, jpt,
                steps > 0 ? joules / (double)steps : 0.0);
    }
    fprintf(out, "    \"generated_tokens\": %d\n", total_tokens);
    fprintf(out, "  }\n");
    fprintf(out, "}\n");
    if (out != stdout) fclose(out);

    printf("train %s %s: %d steps x %d x %d, step %.1f ms, %.1f tok/s, loss %.4f -> %.4f, RSS %.0f MB, GPU busy %.1f%%\n",
           model_dir, precision_name(m->precision), steps, batch_size, seq_len,
           bm_mean(step_ms, steps), bm_mean(token_rate, steps),
           (double)losses[0], (double)losses[steps-1],
           (double)rss_peak / 1e6, busy_pct);
    if (ref_path)
        printf("parity: n=%d max_rel_err=%.9g %s\n", parity_n, parity_max_rel,
               parity_pass ? "PASS" : "FAIL");

    int rc = (check && ref_path && !parity_pass) ? 1 : 0;
    free(ref_losses);
    free(tokens);
    free(inputs);
    free(targets);
    free(step_ms);
    free(token_rate);
    free(losses);
    free(cp_step);
    free(cp_loss);
    bm_destroy_trainer(trainer);
    bm_destroy_model(m);
    bm_destroy(ctx);
    return rc;
}
