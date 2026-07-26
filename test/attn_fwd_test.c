/* attn_fwd_test.c -- validate attention_forward_seq (training forward attention)
 * vs the PyTorch manual causal GQA forward in test/grad/attn_out.bin. cos~1.0. */
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

int main(void) {
    const int NH=4, NKV=2, HD=4, S=3, kv_mul=2; const float scale=0.5f;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float *Q=loadf(G,"attn_Q.bin",NH*S*HD), *K=loadf(G,"attn_K.bin",NKV*S*HD),
          *V=loadf(G,"attn_V.bin",NKV*S*HD), *exp_out=loadf(G,"attn_out.bin",NH*S*HD);
    backend_buffer_t *b_Q=backend_buffer_alloc(be,NH*S*HD*4),*b_K=backend_buffer_alloc(be,NKV*S*HD*4),
                     *b_V=backend_buffer_alloc(be,NKV*S*HD*4),*b_o=backend_buffer_alloc(be,NH*S*HD*4),
                     *b_par=backend_buffer_alloc(be,16*4),*b_sc=backend_buffer_alloc(be,4);
    upload(b_Q,Q,NH*S*HD*4); upload(b_K,K,NKV*S*HD*4); upload(b_V,V,NKV*S*HD*4);
    int par[5]={NH,S,HD,NKV,kv_mul}; upload(b_par,par,sizeof(par));
    upload(b_sc,&scale,sizeof(float));

    backend_kernel_t* k = backend_kernel_create(be, "attention_forward_seq");
    if (!k) { fprintf(stderr, "attention_forward_seq not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_Q,b_K,b_V,b_o,b_par,b_sc};
      backend_encode_dispatch(enc,k,bufs,NULL,6, NH*S,1,1, NH*S,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float* our=backend_buffer_map(b_o); /* no unmap needed, read-only compare */

    double dot=0,na=0,nb=0; float mx=0;
    for (size_t i=0;i<(size_t)NH*S*HD;i++){float a=our[i],b=exp_out[i]; dot+=a*b; na+=a*a; nb+=b*b; float d=fabsf(a-b); if(d>mx)mx=d;}
    double cos=dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("attn_forward_seq out: cos=%.6f maxabs=%.2e\n", cos, mx);
    int ok = cos > 0.9999;
    printf("\n=== attention_forward_seq: %s (cos=%.4f) ===\n", ok?"PASS":"FAIL", cos);

    backend_buffer_unmap(b_o);
    free(Q);free(K);free(V);free(exp_out);
    backend_buffer_free(b_Q);backend_buffer_free(b_K);backend_buffer_free(b_V);
    backend_buffer_free(b_o);backend_buffer_free(b_par);backend_buffer_free(b_sc);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
