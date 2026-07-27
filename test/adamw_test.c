/* adamw_test.c -- validate adamw_step kernel vs torch.optim.AdamW (one step, t=1).
 * In-place update of m, v, w. PASS requires cos ~ 1.0 for all three. */
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
    const int N = 64; const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float* w0 = loadf(G,"adamw_w0.bin",N);
    float* g  = loadf(G,"adamw_g.bin",N);
    float* exp_w = loadf(G,"adamw_w.bin",N);
    float* exp_m = loadf(G,"adamw_m.bin",N);
    float* exp_v = loadf(G,"adamw_v.bin",N);

    backend_buffer_t *b_w=backend_buffer_alloc(be,N*4), *b_g=backend_buffer_alloc(be,N*4),
                     *b_m=backend_buffer_alloc(be,N*4), *b_v=backend_buffer_alloc(be,N*4),
                     *b_n=backend_buffer_alloc(be,4), *b_hp=backend_buffer_alloc(be,28);
    upload(b_w,w0,N*4); upload(b_g,g,N*4);
    memset(backend_buffer_map(b_m),0,N*4); backend_buffer_unmap(b_m);  /* m0=v0=0 */
    memset(backend_buffer_map(b_v),0,N*4); backend_buffer_unmap(b_v);
    upload(b_n,&N,sizeof(int));
    /* lr=0.01 b1=0.9 b2=0.999 eps=1e-8 wd=0.1 bias1=0.1 bias2=0.001 */
    float hp[7] = {0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.1f, 0.001f};
    upload(b_hp,hp,sizeof(hp));

    backend_kernel_t* k = backend_kernel_create(be, "adamw_step");
    if (!k) { fprintf(stderr, "adamw_step not found\n"); return 1; }
    { backend_encoder_t* enc=backend_encode_begin(be);
      backend_buffer_t* bufs[]={b_w,b_g,b_m,b_v,b_n,b_hp};
      backend_encode_dispatch(enc,k,bufs,NULL,6, N,1,1, N,1,1);
      backend_encode_commit(enc); backend_encode_wait(enc); }
    float *our_w=malloc(N*4),*our_m=malloc(N*4),*our_v=malloc(N*4);
    download(b_w,our_w,N*4); download(b_m,our_m,N*4); download(b_v,our_v,N*4);

    float mw,mm,mv;
    double cw=cosine_maxabs(our_w,exp_w,N,&mw);
    double cm=cosine_maxabs(our_m,exp_m,N,&mm);
    double cv=cosine_maxabs(our_v,exp_v,N,&mv);
    printf("w: cos=%.6f maxabs=%.2e\n", cw, mw);
    printf("m: cos=%.6f maxabs=%.2e\n", cm, mm);
    printf("v: cos=%.6f maxabs=%.2e\n", cv, mv);
    int ok = (cw>0.9999)&&(cm>0.9999)&&(cv>0.9999);
    printf("\n=== adamw_step: %s ===\n", ok?"PASS":"FAIL");

    free(w0);free(g);free(exp_w);free(exp_m);free(exp_v);free(our_w);free(our_m);free(our_v);
    backend_buffer_free(b_w);backend_buffer_free(b_g);backend_buffer_free(b_m);
    backend_buffer_free(b_v);backend_buffer_free(b_n);backend_buffer_free(b_hp);
    backend_kernel_destroy(k); backend_destroy(be);
    return ok?0:1;
}
