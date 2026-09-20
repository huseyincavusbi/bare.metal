/* bench/metrics.h -- timing, memory, statistics and probe helpers for the
 * benchmark harness. Header-only so bench.c stays a single translation unit. */
#ifndef BENCH_METRICS_H
#define BENCH_METRICS_H

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

/* Monotonic wall clock in milliseconds. */
static inline double bm_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Peak resident set size in bytes (ru_maxrss is bytes on macOS). */
static inline size_t bm_peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    return (size_t)ru.ru_maxrss;
}

static inline int bm_cmp_dbl(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Nearest-rank percentile in [0,100]. `scratch` is sorted in place. */
static inline double bm_percentile(double* scratch, int n, double p) {
    if (n <= 0) return 0.0;
    qsort(scratch, (size_t)n, sizeof(double), bm_cmp_dbl);
    int idx = (int)((p / 100.0) * (double)(n - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return scratch[idx];
}

/* Percentile on a copy so `src` is left intact. */
static inline double bm_percentile_copy(const double* src, int n, double p) {
    if (n <= 0) return 0.0;
    double* tmp = (double*)malloc((size_t)n * sizeof(double));
    if (!tmp) return 0.0;
    for (int i = 0; i < n; i++) tmp[i] = src[i];
    double r = bm_percentile(tmp, n, p);
    free(tmp);
    return r;
}

static inline double bm_median_copy(const double* src, int n) {
    return bm_percentile_copy(src, n, 50.0);
}

static inline double bm_mean(const double* a, int n) {
    if (n <= 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s / (double)n;
}

/* Population standard deviation. */
static inline double bm_stddev(const double* a, int n) {
    if (n <= 0) return 0.0;
    double mu = bm_mean(a, n);
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - mu; s += d * d; }
    return sqrt(s / (double)n);
}

static inline double bm_min(const double* a, int n) {
    if (n <= 0) return 0.0;
    double v = a[0];
    for (int i = 1; i < n; i++) if (a[i] < v) v = a[i];
    return v;
}

static inline double bm_max(const double* a, int n) {
    if (n <= 0) return 0.0;
    double v = a[0];
    for (int i = 1; i < n; i++) if (a[i] > v) v = a[i];
    return v;
}

static inline unsigned long long bm_sysctl_u64(const char* name) {
    unsigned long long v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0) != 0) return 0;
    return v;
}

static inline void bm_sysctl_str(const char* name, char* buf, size_t n) {
    size_t len = n;
    if (n == 0) return;
    if (sysctlbyname(name, buf, &len, NULL, 0) != 0) buf[0] = '\0';
}

/* Run `cmd`, find the first line containing `needle`, and copy what follows the
 * needle (leading blanks skipped) into `out` (truncated at newline). */
static inline void bm_probe_line(const char* cmd, const char* needle,
                                 char* out, size_t n) {
    if (n == 0) return;
    out[0] = '\0';
    FILE* p = popen(cmd, "r");
    if (!p) return;
    char line[512];
    while (fgets(line, sizeof(line), p)) {
        const char* hit = strstr(line, needle);
        if (hit) {
            hit += strlen(needle);
            while (*hit == ' ' || *hit == '\t') hit++;
            size_t i = 0;
            while (hit[i] && hit[i] != '\n' && hit[i] != '\r' && i + 1 < n) {
                out[i] = hit[i];
                i++;
            }
            out[i] = '\0';
            break;
        }
    }
    pclose(p);
}

/* JSON string escaping for the few free-form fields we emit. */
static inline void bm_json_escape(const char* in, char* out, size_t n) {
    size_t j = 0;
    if (n == 0) return;
    for (size_t i = 0; in && in[i] && j + 2 < n; i++) {
        char c = in[i];
        if (c == '"' || c == '\\') { out[j++] = '\\'; out[j++] = c; }
        else if (c == '\n') { out[j++] = '\\'; out[j++] = 'n'; }
        else out[j++] = c;
    }
    out[j] = '\0';
}

#endif /* BENCH_METRICS_H */
