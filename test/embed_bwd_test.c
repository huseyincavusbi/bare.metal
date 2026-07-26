/* embed_bwd_test.c -- validate embedding_backward vs PyTorch autograd.
 * grad_wte[token[t]] += grad_out[t] (scatter-add, accumulating on repeats).
 * Uses atomic float adds. gwte zeroed first. PASS requires cos ~ 1.0. */
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
    const int V = 8, D = 5, T = 6; const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    int*   tokens = loadi(G, "embed_tokens.bin", T);
    float* gout   = loadf(G, "embed_gout.bin", T*D);
    float* exp_g  = loadf(G, "embed_gwte.bin", V*D);

    backend_buffer_t *b_tok=backend_buffer_alloc(be,T*4), *b_gout=backend_buffer_alloc(be,T*D*4),
                     *b_gw=backend_buffer_alloc(be,V*D*4), *b_d=backend_buffer_alloc(be,4);
    upload(b_tok,tokens,T*4); upload(b_gout,gout,T*D*4); upload(b_d,&D,sizeof(int));
    memset(backend_buffer_map(b_gw),0,V*D*4); backend_buffer_unmap(b_gw);  /* zero grad table */

    backend_kernel_t* k = backend_kernel_create(be, "embedding_backward");
    if (!k) { fprintf(stderr, "embedding_backward not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_tok,b_gout,b_gw,b_d};
      backend_encode_dispatch(enc,k,bufs,NULL,4, T*D,1,1, T*D,1,1);  /* grid = T*D */
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float* our=malloc(V*D*4); download(b_gw,our,V*D*4);

    double dot=0,na=0,nb=0; float mx=0;
    for (size_t i=0;i<(size_t)V*D;i++){float a=our[i],b=exp_g[i]; dot+=a*b; na+=a*a; nb+=b*b; float d=fabsf(a-b); if(d>mx)mx=d;}
    double cos=dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("grad_wte: cos=%.6f maxabs=%.2e\n", cos, mx);
    int ok = cos > 0.9999;
    printf("\n=== embedding_backward: %s (cos=%.4f) ===\n", ok?"PASS":"FAIL", cos);

    free(tokens);free(gout);free(exp_g);free(our);
    backend_buffer_free(b_tok);backend_buffer_free(b_gout);backend_buffer_free(b_gw);backend_buffer_free(b_d);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
