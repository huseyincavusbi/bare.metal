/* matmul_bwd_test.c -- validate matmul_backward_inp + matmul_backward_w
 * against PyTorch autograd reference tensors in test/grad.
 *
 * Forward: out[bt,oc] = bias[oc] + sum_i inp[bt,i]*w[oc,i]
 * Backward wrt input:  ginp[bt,i] = sum_oc gout[bt,oc]*w[oc,i]
 * Backward wrt weight: gw[oc,i]   = sum_bt gout[bt,oc]*inp[bt,i]
 *
 * Loads inp/w/bias/gout (inputs), runs both kernels, compares ginp/gw
 * (our output) to the PyTorch-computed ginp/gw (expected). Reports
 * max abs diff and cosine similarity; PASS requires cos ~ 1.0.
 *
 * Uses only the public backend + registry API, no model/graph. */
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float* load(const char* dir, const char* name, size_t n) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    float* p = malloc(n * sizeof(float));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return p;
}

static void upload(backend_buffer_t* buf, const void* src, size_t bytes) {
    memcpy(backend_buffer_map(buf), src, bytes);
    backend_buffer_unmap(buf);
}

static void download(backend_buffer_t* buf, void* dst, size_t bytes) {
    memcpy(dst, backend_buffer_map(buf), bytes);
    backend_buffer_unmap(buf);
}

int main(void) {
    const int BT = 3, C = 16, OC = 8;
    const char* G = "test/grad";

    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    /* load reference tensors */
    float* inp  = load(G, "inp.bin",  BT*C);
    float* w    = load(G, "w.bin",    OC*C);
    float* gout = load(G, "gout.bin", BT*OC);
    float* exp_ginp = load(G, "ginp.bin", BT*C);
    float* exp_gw   = load(G, "gw.bin",   OC*C);

    /* allocate device buffers */
    backend_buffer_t* b_inp  = backend_buffer_alloc(be, BT*C*sizeof(float));
    backend_buffer_t* b_w    = backend_buffer_alloc(be, OC*C*sizeof(float));
    backend_buffer_t* b_gout = backend_buffer_alloc(be, BT*OC*sizeof(float));
    backend_buffer_t* b_ginp = backend_buffer_alloc(be, BT*C*sizeof(float));
    backend_buffer_t* b_gw   = backend_buffer_alloc(be, OC*C*sizeof(float));
    backend_buffer_t* b_par  = backend_buffer_alloc(be, 16*sizeof(int));

    upload(b_inp, inp, BT*C*sizeof(float));
    upload(b_w,   w,   OC*C*sizeof(float));
    upload(b_gout,gout,BT*OC*sizeof(float));
    /* zero the grad outputs */
    memset(backend_buffer_map(b_ginp), 0, BT*C*sizeof(float)); backend_buffer_unmap(b_ginp);
    memset(backend_buffer_map(b_gw),   0, OC*C*sizeof(float)); backend_buffer_unmap(b_gw);

    int params[3] = {BT, C, OC};
    upload(b_par, params, sizeof(params));

    /* ---- grad_inp: ginp[bt,i] = sum_oc gout[bt,oc]*w[oc,i] ---- */
    backend_kernel_t* k_inp = backend_kernel_create(be, "matmul_backward_inp");
    if (!k_inp) { fprintf(stderr, "kernel matmul_backward_inp not found\n"); return 1; }
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = {b_gout, b_w, b_ginp, b_par};
        /* grid (C, BT), threadgroup (C, BT) -- one thread per element */
        backend_encode_dispatch(enc, k_inp, bufs, NULL, 4, C, BT, 1, C, BT, 1);
        backend_encode_commit(enc);
        backend_encode_wait(enc);
    }

    /* ---- grad_w: gw[oc,i] = sum_bt gout[bt,oc]*inp[bt,i] ---- */
    backend_kernel_t* k_w = backend_kernel_create(be, "matmul_backward_w");
    if (!k_w) { fprintf(stderr, "kernel matmul_backward_w not found\n"); return 1; }
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = {b_gout, b_inp, b_gw, b_par};
        backend_encode_dispatch(enc, k_w, bufs, NULL, 4, C, OC, 1, C, OC, 1);
        backend_encode_commit(enc);
        backend_encode_wait(enc);
    }

    /* download + compare */
    float* our_ginp = malloc(BT*C*sizeof(float));
    float* our_gw   = malloc(OC*C*sizeof(float));
    download(b_ginp, our_ginp, BT*C*sizeof(float));
    download(b_gw,   our_gw,   OC*C*sizeof(float));

    /* cosine + maxabs for both */
    double dot, na, nb; float maxabs;
    dot=na=nb=0; maxabs=0;
    for (size_t i=0;i<(size_t)BT*C;i++){ float a=our_ginp[i],b=exp_ginp[i]; dot+=a*b; na+=a*a; nb+=b*b; float d=fabsf(a-b); if(d>maxabs)maxabs=d; }
    double cos_ginp = dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("grad_inp: cos=%.6f maxabs=%.6f\n", cos_ginp, maxabs);

    dot=na=nb=0; maxabs=0;
    for (size_t i=0;i<(size_t)OC*C;i++){ float a=our_gw[i],b=exp_gw[i]; dot+=a*b; na+=a*a; nb+=b*b; float d=fabsf(a-b); if(d>maxabs)maxabs=d; }
    double cos_gw = dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("grad_w:   cos=%.6f maxabs=%.6f\n", cos_gw, maxabs);

    int ok = (cos_ginp > 0.9999) && (cos_gw > 0.9999);
    printf("\n=== matmul_backward: %s (cos_ginp=%.4f cos_gw=%.4f) ===\n",
           ok ? "PASS" : "FAIL", cos_ginp, cos_gw);

    /* cleanup */
    free(inp); free(w); free(gout); free(exp_ginp); free(exp_gw);
    free(our_ginp); free(our_gw);
    backend_buffer_free(b_inp); backend_buffer_free(b_w); backend_buffer_free(b_gout);
    backend_buffer_free(b_ginp); backend_buffer_free(b_gw); backend_buffer_free(b_par);
    backend_kernel_destroy(k_inp); backend_kernel_destroy(k_w);
    backend_destroy(be);
    return ok ? 0 : 1;
}
