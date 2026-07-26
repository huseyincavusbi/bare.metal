/* rope_bwd_test.c -- validate rope_backward vs PyTorch autograd.
 * RoPE backward is the transposed rotation (orthogonal: R^T = R(-theta)),
 * applied to upstream grads gout_q/gout_k -> gq/gk. No weights.
 * PASS requires cos ~ 1.0 for both q and k gradients. */
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
    const int NH = 6, NKV = 2, HD = 8, pos = 3; const float theta = 10000.0f;
    const char* G = "test/grad";

    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float* gout_q = load(G, "rope_gout_q.bin", NH*HD);
    float* gout_k = load(G, "rope_gout_k.bin", NKV*HD);
    float* exp_gq = load(G, "rope_gq.bin", NH*HD);
    float* exp_gk = load(G, "rope_gk.bin", NKV*HD);

    backend_buffer_t *b_gout_q=backend_buffer_alloc(be,NH*HD*4), *b_gout_k=backend_buffer_alloc(be,NKV*HD*4),
                     *b_gq=backend_buffer_alloc(be,NH*HD*4),      *b_gk=backend_buffer_alloc(be,NKV*HD*4),
                     *b_hd=backend_buffer_alloc(be,4), *b_pos=backend_buffer_alloc(be,4),
                     *b_theta=backend_buffer_alloc(be,4), *b_nkv=backend_buffer_alloc(be,4);
    upload(b_gout_q,gout_q,NH*HD*4); upload(b_gout_k,gout_k,NKV*HD*4);

    upload(b_hd,&HD,sizeof(int)); upload(b_pos,&pos,sizeof(int));
    upload(b_theta,&theta,sizeof(float)); upload(b_nkv,&NKV,sizeof(int));

    backend_kernel_t* k = backend_kernel_create(be, "rope_backward");
    if (!k) { fprintf(stderr, "kernel rope_backward not found\n"); return 1; }
    {
        backend_encoder_t* enc = backend_encode_begin(be);
        backend_buffer_t* bufs[] = {b_gout_q, b_gout_k, b_gq, b_gk, b_hd, b_pos, b_theta, b_nkv};
        backend_encode_dispatch(enc, k, bufs, NULL, 8, NH,1,1, NH,1,1);  /* grid = NH */
        backend_encode_commit(enc); backend_encode_wait(enc);
    }

    float *our_gq = malloc(NH*HD*4), *our_gk = malloc(NKV*HD*4);
    download(b_gq, our_gq, NH*HD*4); download(b_gk, our_gk, NKV*HD*4);

    float mq, mk;
    double cos_gq = cosine_maxabs(our_gq, exp_gq, NH*HD, &mq);
    double cos_gk = cosine_maxabs(our_gk, exp_gk, NKV*HD, &mk);
    printf("grad_q: cos=%.6f maxabs=%.6f\n", cos_gq, mq);
    printf("grad_k: cos=%.6f maxabs=%.6f\n", cos_gk, mk);
    int ok = (cos_gq > 0.9999) && (cos_gk > 0.9999);
    printf("\n=== rope_backward: %s (cos_gq=%.4f cos_gk=%.4f) ===\n",
           ok ? "PASS" : "FAIL", cos_gq, cos_gk);

    free(gout_q);free(gout_k);free(exp_gq);free(exp_gk);free(our_gq);free(our_gk);
    backend_buffer_free(b_gout_q);backend_buffer_free(b_gout_k);backend_buffer_free(b_gq);
    backend_buffer_free(b_gk);backend_buffer_free(b_hd);backend_buffer_free(b_pos);
    backend_buffer_free(b_theta);backend_buffer_free(b_nkv);
    backend_kernel_destroy(k);
    backend_destroy(be);
    return ok ? 0 : 1;
}
