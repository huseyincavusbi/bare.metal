/* attn_bwd_test.c -- validate attention_backward (causal+GQA) vs PyTorch autograd.
 * One thread per (h,i). gradK/gradV zeroed, atomic-accumulated. cos~1.0 for all. */
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
    const int NH=4, NKV=2, HD=4, S=3, kv_mul=2; const float scale=0.5f;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float *Q=loadf(G,"attn_Q.bin",NH*S*HD), *K=loadf(G,"attn_K.bin",NKV*S*HD),
          *V=loadf(G,"attn_V.bin",NKV*S*HD), *gout=loadf(G,"attn_gout.bin",NH*S*HD);
    float *exp_gQ=loadf(G,"attn_gQ.bin",NH*S*HD), *exp_gK=loadf(G,"attn_gK.bin",NKV*S*HD),
          *exp_gV=loadf(G,"attn_gV.bin",NKV*S*HD);

    backend_buffer_t *b_gout=backend_buffer_alloc(be,NH*S*HD*4), *b_Q=backend_buffer_alloc(be,NH*S*HD*4),
                     *b_K=backend_buffer_alloc(be,NKV*S*HD*4),    *b_V=backend_buffer_alloc(be,NKV*S*HD*4),
                     *b_gQ=backend_buffer_alloc(be,NH*S*HD*4),    *b_gK=backend_buffer_alloc(be,NKV*S*HD*4),
                     *b_gV=backend_buffer_alloc(be,NKV*S*HD*4),   *b_par=backend_buffer_alloc(be,16*4),
                     *b_sc=backend_buffer_alloc(be,4);
    upload(b_gout,gout,NH*S*HD*4); upload(b_Q,Q,NH*S*HD*4); upload(b_K,K,NKV*S*HD*4); upload(b_V,V,NKV*S*HD*4);
    memset(backend_buffer_map(b_gK),0,NKV*S*HD*4); backend_buffer_unmap(b_gK);
    memset(backend_buffer_map(b_gV),0,NKV*S*HD*4); backend_buffer_unmap(b_gV);
    int par[5]={NH,S,HD,NKV,kv_mul}; upload(b_par,par,sizeof(par));
    upload(b_sc,&scale,sizeof(float));

    backend_kernel_t* k = backend_kernel_create(be, "attention_backward");
    if (!k) { fprintf(stderr, "attention_backward not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_gout,b_Q,b_K,b_V,b_gQ,b_gK,b_gV,b_par,b_sc};
      backend_encode_dispatch(enc,k,bufs,NULL,9, NH*S,1,1, NH*S,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float *our_gQ=malloc(NH*S*HD*4),*our_gK=malloc(NKV*S*HD*4),*our_gV=malloc(NKV*S*HD*4);
    download(b_gQ,our_gQ,NH*S*HD*4); download(b_gK,our_gK,NKV*S*HD*4); download(b_gV,our_gV,NKV*S*HD*4);

    float mq,mk,mv;
    double cq=cosine_maxabs(our_gQ,exp_gQ,NH*S*HD,&mq);
    double ck=cosine_maxabs(our_gK,exp_gK,NKV*S*HD,&mk);
    double cv=cosine_maxabs(our_gV,exp_gV,NKV*S*HD,&mv);
    printf("grad_Q: cos=%.6f maxabs=%.2e\n", cq, mq);
    printf("grad_K: cos=%.6f maxabs=%.2e\n", ck, mk);
    printf("grad_V: cos=%.6f maxabs=%.2e\n", cv, mv);
    int ok = (cq>0.9999)&&(ck>0.9999)&&(cv>0.9999);
    printf("\n=== attention_backward: %s ===\n", ok?"PASS":"FAIL");

    free(Q);free(K);free(V);free(gout);free(exp_gQ);free(exp_gK);free(exp_gV);
    free(our_gQ);free(our_gK);free(our_gV);
    backend_buffer_free(b_gout);backend_buffer_free(b_Q);backend_buffer_free(b_K);backend_buffer_free(b_V);
    backend_buffer_free(b_gQ);backend_buffer_free(b_gK);backend_buffer_free(b_gV);
    backend_buffer_free(b_par);backend_buffer_free(b_sc);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
