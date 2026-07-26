/* act_bwd_test.c -- validate gelu_backward + swiglu_backward vs PyTorch autograd.
 * Both are elementwise. gelu: 1 input -> grad_x. swiglu: gate,up -> grad_gate,grad_up.
 * PASS requires cos ~ 1.0 for all gradients. */
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
    const int N = 32; const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }
    int fails = 0;

    /* ---- GELU backward ---- */
    float *gx_in = load(G,"gelu_x.bin",N), *gout_g = load(G,"gelu_gout.bin",N), *exp_gx = load(G,"gelu_gx.bin",N);
    backend_buffer_t *b_x=backend_buffer_alloc(be,N*4),*b_gout=backend_buffer_alloc(be,N*4),
                     *b_gx=backend_buffer_alloc(be,N*4),*b_n=backend_buffer_alloc(be,4);
    upload(b_x,gx_in,N*4); upload(b_gout,gout_g,N*4); upload(b_n,&N,sizeof(int));
    backend_kernel_t* kg = backend_kernel_create(be,"gelu_backward");
    if (!kg) { fprintf(stderr,"gelu_backward not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_gout,b_x,b_gx,b_n};
      backend_encode_dispatch(enc,kg,bufs,NULL,4, N,1,1, N,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float *our_gx=malloc(N*4); download(b_gx,our_gx,N*4);
    float mg; double cg=cosine_maxabs(our_gx,exp_gx,N,&mg);
    printf("gelu grad_x: cos=%.6f maxabs=%.2e\n", cg, mg);
    if (cg<=0.9999) fails++;

    /* ---- SwiGLU backward ---- */
    float *gate=load(G,"swiglu_gate.bin",N),*up=load(G,"swiglu_up.bin",N),
          *gout_s=load(G,"swiglu_gout.bin",N),*exp_ggate=load(G,"swiglu_ggate.bin",N),*exp_gup=load(G,"swiglu_gup.bin",N);
    backend_buffer_t *b_gate=backend_buffer_alloc(be,N*4),*b_up=backend_buffer_alloc(be,N*4),
                     *b_gout_s=backend_buffer_alloc(be,N*4),*b_ggate=backend_buffer_alloc(be,N*4),
                     *b_gup=backend_buffer_alloc(be,N*4);
    upload(b_gate,gate,N*4); upload(b_up,up,N*4); upload(b_gout_s,gout_s,N*4);
    backend_kernel_t* ks = backend_kernel_create(be,"swiglu_backward");
    if (!ks) { fprintf(stderr,"swiglu_backward not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_gate,b_up,b_gout_s,b_ggate,b_gup,b_n};
      backend_encode_dispatch(enc,ks,bufs,NULL,6, N,1,1, N,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float *our_ggate=malloc(N*4),*our_gup=malloc(N*4);
    download(b_ggate,our_ggate,N*4); download(b_gup,our_gup,N*4);
    float mgg,mgu; double cgg=cosine_maxabs(our_ggate,exp_ggate,N,&mgg);
    double cgu=cosine_maxabs(our_gup,exp_gup,N,&mgu);
    printf("swiglu grad_gate: cos=%.6f maxabs=%.2e\n", cgg, mgg);
    printf("swiglu grad_up:   cos=%.6f maxabs=%.2e\n", cgu, mgu);
    if (cgg<=0.9999||cgu<=0.9999) fails++;

    int ok = (fails==0);
    printf("\n=== act_backward (gelu+swiglu): %s ===\n", ok?"PASS":"FAIL");

    free(gx_in);free(gout_g);free(exp_gx);free(our_gx);
    free(gate);free(up);free(gout_s);free(exp_ggate);free(exp_gup);free(our_ggate);free(our_gup);
    backend_buffer_free(b_x);backend_buffer_free(b_gout);backend_buffer_free(b_gx);backend_buffer_free(b_n);
    backend_buffer_free(b_gate);backend_buffer_free(b_up);backend_buffer_free(b_gout_s);
    backend_buffer_free(b_ggate);backend_buffer_free(b_gup);
    backend_kernel_destroy(kg);backend_kernel_destroy(ks);
    backend_destroy(be);
    return ok?0:1;
}
