/* rmsnorm_bwd_test.c -- validate rmsnorm_backward_x + rmsnorm_backward_w
 * against PyTorch autograd reference tensors in test/grad/rms_*.bin.
 *
 * Forward: ss = sum(x^2)/C ; inv_rms = 1/sqrt(ss+eps) ; out[j] = x[j]*inv_rms*w[j]
 * Backward wrt input:  gx[j] = inv_rms*(gout[j]*w[j] - n[j]*c1), c1=(1/C)*sum gout*out
 * Backward wrt weight: gw[j] = sum_row gout[row,j]*n[row,j]
 *
 * PASS requires cos ~ 1.0 for both gradients. Uses only the public backend API. */
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float* load(const char* dir, const char* name, size_t n) {
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    float* p = malloc(n * sizeof(float));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return p;
}
static void upload(backend_buffer_t* buf, const void* src, size_t bytes) {
    memcpy(backend_buffer_map(buf), src, bytes); backend_buffer_unmap(buf);
}
static void download(backend_buffer_t* buf, void* dst, size_t bytes) {
    memcpy(dst, backend_buffer_map(buf), bytes); backend_buffer_unmap(buf);
}
static double cosine_maxabs(const float* a, const float* b, size_t n, float* maxabs) {
    double dot=0, na=0, nb=0; *maxabs=0;
    for (size_t i=0;i<n;i++){ float x=a[i],y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; float d=fabsf(x-y); if(d>*maxabs)*maxabs=d; }
    return dot/(sqrt(na)*sqrt(nb)+1e-12);
}

int main(void) {
    const int N = 4, C = 24; const float eps = 1e-5f;
    const char* G = "test/grad";

    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float* x   = load(G, "rms_x.bin",   N*C);
    float* w   = load(G, "rms_w.bin",   C);
    float* gout= load(G, "rms_gout.bin",N*C);
    float* exp_gx = load(G, "rms_gx.bin", N*C);
    float* exp_gw = load(G, "rms_gw.bin", C);

    backend_buffer_t *b_gout=backend_buffer_alloc(be,N*C*4), *b_w=backend_buffer_alloc(be,C*4),
                     *b_x=backend_buffer_alloc(be,N*C*4),    *b_gx=backend_buffer_alloc(be,N*C*4),
                     *b_gw=backend_buffer_alloc(be,C*4),     *b_par=backend_buffer_alloc(be,16*4),
                     *b_eps=backend_buffer_alloc(be,4);
    upload(b_gout,gout,N*C*4); upload(b_w,w,C*4); upload(b_x,x,N*C*4);
    memset(backend_buffer_map(b_gx),0,N*C*4); backend_buffer_unmap(b_gx);
    memset(backend_buffer_map(b_gw),0,C*4);   backend_buffer_unmap(b_gw);
    int par[2]={N,C}; upload(b_par,par,sizeof(par));
    upload(b_eps,&eps,sizeof(float));

    /* grad_x: one thread per row, grid (N,) */
    backend_kernel_t* kx = backend_kernel_create(be, "rmsnorm_backward_x");
    if (!kx) { fprintf(stderr, "kernel rmsnorm_backward_x not found\n"); return 1; }
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = {b_gout, b_w, b_x, b_gx, b_par, b_eps};
        backend_encode_dispatch(enc, kx, bufs, NULL, 6, N,1,1, N,1,1);
        backend_encode_commit(enc); backend_encode_wait(enc);
    }
    /* grad_w: one thread per column, grid (C,) */
    backend_kernel_t* kw = backend_kernel_create(be, "rmsnorm_backward_w");
    if (!kw) { fprintf(stderr, "kernel rmsnorm_backward_w not found\n"); return 1; }
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = {b_gout, b_x, b_gw, b_par, b_eps};
        backend_encode_dispatch(enc, kw, bufs, NULL, 5, C,1,1, C,1,1);
        backend_encode_commit(enc); backend_encode_wait(enc);
    }

    float *our_gx = malloc(N*C*4), *our_gw = malloc(C*4);
    download(b_gx, our_gx, N*C*4); download(b_gw, our_gw, C*4);

    float mx, mw;
    double cos_gx = cosine_maxabs(our_gx, exp_gx, N*C, &mx);
    double cos_gw = cosine_maxabs(our_gw, exp_gw, C, &mw);
    printf("grad_x: cos=%.6f maxabs=%.6f\n", cos_gx, mx);
    printf("grad_w: cos=%.6f maxabs=%.6f\n", cos_gw, mw);
    int ok = (cos_gx > 0.9999) && (cos_gw > 0.9999);
    printf("\n=== rmsnorm_backward: %s (cos_gx=%.4f cos_gw=%.4f) ===\n",
           ok ? "PASS" : "FAIL", cos_gx, cos_gw);

    free(x);free(w);free(gout);free(exp_gx);free(exp_gw);free(our_gx);free(our_gw);
    backend_buffer_free(b_gout);backend_buffer_free(b_w);backend_buffer_free(b_x);
    backend_buffer_free(b_gx);backend_buffer_free(b_gw);backend_buffer_free(b_par);backend_buffer_free(b_eps);
    backend_kernel_destroy(kx); backend_kernel_destroy(kw);
    backend_destroy(be);
    return ok ? 0 : 1;
}
