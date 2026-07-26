/* xent_bwd_test.c -- validate xent_backward (cross-entropy loss grad) vs autograd.
 * grad_logits[n,j] = (softmax(logits[n])[j] - onehot(target[n])) / N
 * PASS requires cos ~ 1.0. */
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float* loadf(const char* dir, const char* name, size_t n) {
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    float* p = malloc(n * sizeof(float));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return p;
}
static int* loadi(const char* dir, const char* name, size_t n) {
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, name);
    int* p = malloc(n * sizeof(int));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(p, sizeof(int), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return p;
}
static void upload(backend_buffer_t* buf, const void* src, size_t bytes) {
    memcpy(backend_buffer_map(buf), src, bytes); backend_buffer_unmap(buf);
}
static void download(backend_buffer_t* buf, void* dst, size_t bytes) {
    memcpy(dst, backend_buffer_map(buf), bytes); backend_buffer_unmap(buf);
}

int main(void) {
    const int N = 3, V = 10; const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float* logits = loadf(G, "xent_logits.bin", N*V);
    int*   targets= loadi(G, "xent_targets.bin", N);
    float* exp_g  = loadf(G, "xent_glogits.bin", N*V);

    backend_buffer_t *b_l=backend_buffer_alloc(be,N*V*4), *b_t=backend_buffer_alloc(be,N*4),
                     *b_g=backend_buffer_alloc(be,N*V*4), *b_p=backend_buffer_alloc(be,16*4);
    upload(b_l,logits,N*V*4); upload(b_t,targets,N*4);
    int par[2]={N,V}; upload(b_p,par,sizeof(par));

    backend_kernel_t* k = backend_kernel_create(be, "xent_backward");
    if (!k) { fprintf(stderr, "xent_backward not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_l,b_t,b_g,b_p};
      backend_encode_dispatch(enc,k,bufs,NULL,4, N,1,1, N,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float* our=malloc(N*V*4); download(b_g,our,N*V*4);

    double dot=0,na=0,nb=0; float mx=0;
    for (size_t i=0;i<(size_t)N*V;i++){float a=our[i],b=exp_g[i]; dot+=a*b; na+=a*a; nb+=b*b; float d=fabsf(a-b); if(d>mx)mx=d;}
    double cos=dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("grad_logits: cos=%.6f maxabs=%.2e\n", cos, mx);
    int ok = cos > 0.9999;
    printf("\n=== xent_backward: %s (cos=%.4f) ===\n", ok?"PASS":"FAIL", cos);

    free(logits);free(targets);free(exp_g);free(our);
    backend_buffer_free(b_l);backend_buffer_free(b_t);backend_buffer_free(b_g);backend_buffer_free(b_p);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
