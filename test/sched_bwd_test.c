/* sched_bwd_test.c -- G2 sub-graph: scheduler forward + backward walk on a
 * hand-built 3-node graph (matmul -> rmsnorm -> matmul), validated vs autograd.
 *
 * Proves the symphony: the SAME scheduler that does the forward walk also does
 * the backward walk (reverse topo, grad buffers, per-op backward dispatch).
 * Compares grad_inp, grad_w1, grad_rmsw, grad_w2 to PyTorch. cos~1.0 = PASS. */
#include "baremetal/graph.h"
#include "baremetal/scheduler.h"
#include "kernels/registry.h"
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
    const int BT=4, C1=16, OC1=8, OC2=5; const float eps=1e-5f;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float *inp =loadf(G,"sb_inp.bin", BT*C1),  *w1 =loadf(G,"sb_w1.bin", OC1*C1),
          *rmsw=loadf(G,"sb_rmsw.bin", OC1),    *w2 =loadf(G,"sb_w2.bin", OC2*OC1),
          *gout=loadf(G,"sb_gout.bin", BT*OC2);
    float *exp_ginp =loadf(G,"sb_ginp.bin", BT*C1),
          *exp_gw1 =loadf(G,"sb_gw1.bin", OC1*C1),
          *exp_grmsw=loadf(G,"sb_grmsw.bin", OC1),
          *exp_gw2 =loadf(G,"sb_gw2.bin", OC2*OC1);

    /* build graph: inp[4,16] -> matmul(w1[8,16]) -> [4,8] -> rmsnorm(rmsw[8]) -> [4,8] -> matmul(w2[5,8]) -> [4,5] */
    bmt_graph_t* g = bmt_graph_create();
    int t0 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){BT, C1});
    int t1 = bmt_graph_add_weight(g, w1, 2, (int[]){OC1, C1});
    int t2 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){BT, OC1});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t0, t1}, t2, 3, (int[]){BT, C1, OC1}, 0, NULL);
    int t3 = bmt_graph_add_weight(g, rmsw, 1, (int[]){OC1});
    int t4 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){BT, OC1});
    bmt_graph_add_node(g, BMK_OP_NORM_RMS, 2, (int[]){t2, t3}, t4, 2, (int[]){BT, OC1}, 1, (float[]){eps});
    int t5 = bmt_graph_add_weight(g, w2, 2, (int[]){OC2, OC1});
    int t6 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){BT, OC2});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t4, t5}, t6, 3, (int[]){BT, OC1, OC2}, 0, NULL);

    /* registry + scheduler */
    bmk_registry_t* reg = bmk_registry_create(be);
    bmk_register(reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmt_scheduler_t* sched = bmt_scheduler_create(be, reg, g, BT, OC1, 1);

    /* forward: set input, run, get output */
    bmt_scheduler_set_input(sched, t0, inp, BT*C1*sizeof(float));
    bmt_scheduler_run(sched, 0, BT);

    /* forward stage checks: t2=h, t4=hn, t6=out */
    float *fwd_h=malloc(BT*OC1*4), *fwd_hn=malloc(BT*OC1*4), *fwd_out=malloc(BT*OC2*4);
    bmt_scheduler_get_output(sched, t2, fwd_h, BT*OC1*4);
    bmt_scheduler_get_output(sched, t4, fwd_hn, BT*OC1*4);
    bmt_scheduler_get_output(sched, t6, fwd_out, BT*OC2*4);
    float *exp_h=loadf(G,"sb_h.bin",BT*OC1), *exp_hn=loadf(G,"sb_hn.bin",BT*OC1), *exp_out=loadf(G,"sb_out.bin",BT*OC2);
    { float m; double c=cosine_maxabs(fwd_h,exp_h,BT*OC1,&m); printf("FWD matmul1  out: cos=%.6f maxabs=%.2e\n",c,m); }
    { float m; double c=cosine_maxabs(fwd_hn,exp_hn,BT*OC1,&m); printf("FWD rmsnorm  out: cos=%.6f maxabs=%.2e\n",c,m); }
    { float m; double c=cosine_maxabs(fwd_out,exp_out,BT*OC2,&m); printf("FWD matmul2  out: cos=%.6f maxabs=%.2e\n",c,m); }
    free(fwd_h);free(fwd_hn);free(fwd_out);free(exp_h);free(exp_hn);free(exp_out);

    /* seed grad of final output, then backward */
    bmt_scheduler_set_grad(sched, t6, gout, BT*OC2*sizeof(float));
    bmt_scheduler_backward(sched);

    /* read grads */
    float *ginp=malloc(BT*C1*4), *gw1=malloc(OC1*C1*4), *grmsw=malloc(OC1*4), *gw2=malloc(OC2*OC1*4);
    bmt_scheduler_get_grad(sched, t0, ginp, BT*C1*4);
    bmt_scheduler_get_grad(sched, t1, gw1, OC1*C1*4);
    bmt_scheduler_get_grad(sched, t3, grmsw, OC1*4);
    bmt_scheduler_get_grad(sched, t5, gw2, OC2*OC1*4);

    float m1,m2,m3,m4;
    double c1=cosine_maxabs(ginp, exp_ginp, BT*C1, &m1);
    double c2=cosine_maxabs(gw1, exp_gw1, OC1*C1, &m2);
    double c3=cosine_maxabs(grmsw, exp_grmsw, OC1, &m3);
    double c4=cosine_maxabs(gw2, exp_gw2, OC2*OC1, &m4);
    printf("grad_inp : cos=%.6f maxabs=%.2e\n", c1, m1);
    printf("grad_w1  : cos=%.6f maxabs=%.2e\n", c2, m2);
    printf("grad_rmsw: cos=%.6f maxabs=%.2e\n", c3, m3);
    printf("grad_w2  : cos=%.6f maxabs=%.2e\n", c4, m4);
    int ok = (c1>0.9999)&&(c2>0.9999)&&(c3>0.9999)&&(c4>0.9999);
    printf("\n=== scheduler backward walk (matmul->rmsnorm->matmul): %s ===\n", ok?"PASS":"FAIL");

    free(inp);free(w1);free(rmsw);free(w2);free(gout);
    free(exp_ginp);free(exp_gw1);free(exp_grmsw);free(exp_gw2);
    free(ginp);free(gw1);free(grmsw);free(gw2);
    bmt_scheduler_destroy(sched); bmk_registry_destroy(reg); bmt_graph_destroy(g);
    backend_destroy(be);
    return ok?0:1;
}
