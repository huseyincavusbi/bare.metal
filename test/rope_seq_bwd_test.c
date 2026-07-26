/* rope_seq_bwd_test.c -- validate rope_backward_seq (per-position, [S,NH,HD])
 * vs PyTorch autograd. Transposed rotation by pos=s. cos~1.0 for gq and gk. */
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
static double cosine_maxabs(const float* a, const float* b, size_t n, float* maxabs) {
    double dot=0, na=0, nb=0; *maxabs=0;
    for (size_t i=0;i<n;i++){ float x=a[i],y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; float d=fabsf(x-y); if(d>*maxabs)*maxabs=d; }
    return dot/(sqrt(na)*sqrt(nb)+1e-12);
}

int main(void) {
    const int NH=4, NKV=2, HD=8, S=5; const float theta=10000.0f;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float *gout_q=loadf(G,"roseq_gout_q.bin",S*NH*HD), *gout_k=loadf(G,"roseq_gout_k.bin",S*NKV*HD);
    float *exp_gQ=loadf(G,"roseq_gQ.bin",S*NH*HD), *exp_gK=loadf(G,"roseq_gK.bin",S*NKV*HD);

    backend_buffer_t *b_gq=backend_buffer_alloc(be,S*NH*HD*4), *b_gk=backend_buffer_alloc(be,S*NKV*HD*4),
                     *b_oq=backend_buffer_alloc(be,S*NH*HD*4), *b_ok=backend_buffer_alloc(be,S*NKV*HD*4),
                     *b_par=backend_buffer_alloc(be,16*4), *b_th=backend_buffer_alloc(be,4);
    memcpy(backend_buffer_map(b_gq),gout_q,S*NH*HD*4); backend_buffer_unmap(b_gq);
    memcpy(backend_buffer_map(b_gk),gout_k,S*NKV*HD*4); backend_buffer_unmap(b_gk);
    int par[4]={HD,NKV,S,NH}; memcpy(backend_buffer_map(b_par),par,sizeof(par)); backend_buffer_unmap(b_par);
    memcpy(backend_buffer_map(b_th),&theta,sizeof(float)); backend_buffer_unmap(b_th);

    backend_kernel_t* k = backend_kernel_create(be, "rope_backward_seq");
    if (!k) { fprintf(stderr, "rope_backward_seq not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_gq,b_gk,b_oq,b_ok,b_par,b_th};
      backend_encode_dispatch(enc,k,bufs,NULL,6, S*NH,1,1, S*NH,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float *our_gQ=backend_buffer_map(b_oq), *our_gK=backend_buffer_map(b_ok);

    float mq,mk;
    double cq=cosine_maxabs(our_gQ,exp_gQ,S*NH*HD,&mq);
    double ck=cosine_maxabs(our_gK,exp_gK,S*NKV*HD,&mk);
    printf("grad_q: cos=%.6f maxabs=%.2e\n", cq, mq);
    printf("grad_k: cos=%.6f maxabs=%.2e\n", ck, mk);
    int ok = (cq>0.9999)&&(ck>0.9999);
    printf("\n=== rope_backward_seq: %s ===\n", ok?"PASS":"FAIL");

    backend_buffer_unmap(b_oq); backend_buffer_unmap(b_ok);
    free(gout_q);free(gout_k);free(exp_gQ);free(exp_gK);
    backend_buffer_free(b_gq);backend_buffer_free(b_gk);backend_buffer_free(b_oq);
    backend_buffer_free(b_ok);backend_buffer_free(b_par);backend_buffer_free(b_th);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
