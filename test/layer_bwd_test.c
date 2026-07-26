/* layer_bwd_test.c -- Full single layer forward + backward G2 test
 * Tests the entire layer graph: norm1 -> Q/K/V -> RoPE -> attn -> proj -> ADD1 -> norm2 -> FFN -> ADD2
 */
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
    const int S=8, D=64, NH=4, NKV=1, HD=16, HID=128;
    const int KV = NKV * HD;
    const float eps=1e-5f, theta=10000.0f;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    float *inp = loadf(G,"layer_inp.bin", S*D),
          *rmsw1 = loadf(G,"layer_rmsw1.bin", D),
          *wq = loadf(G,"layer_wq.bin", D*D),
          *wk = loadf(G,"layer_wk.bin", KV*D),
          *wv = loadf(G,"layer_wv.bin", KV*D),
          *wproj = loadf(G,"layer_wproj.bin", D*D),
          *rmsw2 = loadf(G,"layer_rmsw2.bin", D),
          *wgate = loadf(G,"layer_wgate.bin", HID*D),
          *wup = loadf(G,"layer_wup.bin", HID*D),
          *wdown = loadf(G,"layer_wdown.bin", D*HID),
          *gout = loadf(G,"layer_gout.bin", S*D);

    float *exp_ginp = loadf(G,"layer_ginp.bin", S*D),
          *exp_grmsw1 = loadf(G,"layer_grmsw1.bin", D),
          *exp_gwq = loadf(G,"layer_gwq.bin", D*D),
          *exp_gwk = loadf(G,"layer_gwk.bin", KV*D),
          *exp_gwv = loadf(G,"layer_gwv.bin", KV*D),
          *exp_gwproj = loadf(G,"layer_gwproj.bin", D*D),
          *exp_grmsw2 = loadf(G,"layer_grmsw2.bin", D),
          *exp_gwgate = loadf(G,"layer_gwgate.bin", HID*D),
          *exp_gwup = loadf(G,"layer_gwup.bin", HID*D),
          *exp_gwdown = loadf(G,"layer_gwdown.bin", D*HID);

    bmt_graph_t* g = bmt_graph_create();
    
    // 0. Input
    int t_inp = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    
    // 1. Norm 1
    int t_rmsw1 = bmt_graph_add_weight(g, rmsw1, 1, (int[]){D});
    int t_n1 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_NORM_RMS, 2, (int[]){t_inp, t_rmsw1}, t_n1, 2, (int[]){S, D}, 1, (float[]){eps});

    // 2. Q/K/V
    int t_wq = bmt_graph_add_weight(g, wq, 2, (int[]){D, D});
    int t_wk = bmt_graph_add_weight(g, wk, 2, (int[]){KV, D});
    int t_wv = bmt_graph_add_weight(g, wv, 2, (int[]){KV, D});
    int t_q = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    int t_k = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, KV});
    int t_v = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, KV});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_n1, t_wq}, t_q, 3, (int[]){S, D, D}, 0, NULL);
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_n1, t_wk}, t_k, 3, (int[]){S, D, KV}, 0, NULL);
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_n1, t_wv}, t_v, 3, (int[]){S, D, KV}, 0, NULL);

    // 3. RoPE (in-place on Q, K)
    bmt_graph_add_node(g, BMK_OP_POS_ENC_ROPE, 2, (int[]){t_q, t_k}, t_q, 3, (int[]){HD, NKV, NH}, 1, (float[]){theta});

    // 4. Attention
    int t_attn_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_ATTENTION, 3, (int[]){t_q, t_k, t_v}, t_attn_out, 8, (int[]){NH, HD, NKV, KV, 1, NH/NKV, 0, S}, 0, NULL);

    // 5. Projection
    int t_wproj = bmt_graph_add_weight(g, wproj, 2, (int[]){D, D});
    int t_proj_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_attn_out, t_wproj}, t_proj_out, 3, (int[]){S, D, D}, 0, NULL);

    // 6. ADD 1
    int t_res1 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_ADD, 2, (int[]){t_inp, t_proj_out}, t_res1, 1, (int[]){S*D}, 0, NULL);

    // 7. Norm 2
    int t_rmsw2 = bmt_graph_add_weight(g, rmsw2, 1, (int[]){D});
    int t_n2 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_NORM_RMS, 2, (int[]){t_res1, t_rmsw2}, t_n2, 2, (int[]){S, D}, 1, (float[]){eps});

    // 8. FFN (SwiGLU)
    int t_wgate = bmt_graph_add_weight(g, wgate, 2, (int[]){HID, D});
    int t_wup = bmt_graph_add_weight(g, wup, 2, (int[]){HID, D});
    int t_wdown = bmt_graph_add_weight(g, wdown, 2, (int[]){D, HID});
    
    int t_gate = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, HID});
    int t_up = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, HID});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_n2, t_wgate}, t_gate, 3, (int[]){S, D, HID}, 0, NULL);
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_n2, t_wup}, t_up, 3, (int[]){S, D, HID}, 0, NULL);
    
    int t_silu = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, HID});
    bmt_graph_add_node(g, BMK_OP_ACT_SWIGLU, 2, (int[]){t_gate, t_up}, t_silu, 1, (int[]){S*HID}, 0, NULL);
    
    int t_down_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_silu, t_wdown}, t_down_out, 3, (int[]){S, HID, D}, 0, NULL);

    // 9. ADD 2
    int t_res2 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, (int[]){S, D});
    bmt_graph_add_node(g, BMK_OP_ADD, 2, (int[]){t_res1, t_down_out}, t_res2, 1, (int[]){S*D}, 0, NULL);

    bmk_registry_t* reg = bmk_registry_create(be);
    bmk_register(reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");

    bmt_scheduler_t* sched = bmt_scheduler_create(be, reg, g, S, KV, 1);

    bmt_scheduler_set_input(sched, t_inp, inp, S*D*sizeof(float));
    bmt_scheduler_forward_train(sched, S);

    // Read and check forward intermediates
    float *fwd_n1=malloc(S*D*4), *fwd_q=malloc(S*D*4), *fwd_k=malloc(S*KV*4), *fwd_attn_out=malloc(S*D*4),
          *fwd_res1=malloc(S*D*4), *fwd_n2=malloc(S*D*4), *fwd_res2=malloc(S*D*4);
    bmt_scheduler_get_output(sched, t_n1, fwd_n1, S*D*4);
    bmt_scheduler_get_output(sched, t_q, fwd_q, S*D*4);
    bmt_scheduler_get_output(sched, t_k, fwd_k, S*KV*4);
    bmt_scheduler_get_output(sched, t_attn_out, fwd_attn_out, S*D*4);
    bmt_scheduler_get_output(sched, t_res1, fwd_res1, S*D*4);
    bmt_scheduler_get_output(sched, t_n2, fwd_n2, S*D*4);
    bmt_scheduler_get_output(sched, t_res2, fwd_res2, S*D*4);

    float *exp_n1=loadf(G,"layer_n1.bin",S*D), *exp_q=loadf(G,"layer_q.bin",S*D), *exp_k=loadf(G,"layer_k.bin",S*KV),
          *exp_attn_out=loadf(G,"layer_attn_out.bin",S*D), *exp_res1=loadf(G,"layer_res1.bin",S*D),
          *exp_n2=loadf(G,"layer_n2.bin",S*D), *exp_res2=loadf(G,"layer_res2.bin",S*D);
    float mf;
    printf("FWD n1       : cos=%.6f\n", cosine_maxabs(fwd_n1, exp_n1, S*D, &mf));
    printf("FWD q (rope) : cos=%.6f\n", cosine_maxabs(fwd_q, exp_q, S*D, &mf));
    printf("FWD k (rope) : cos=%.6f\n", cosine_maxabs(fwd_k, exp_k, S*KV, &mf));
    printf("FWD attn_out : cos=%.6f\n", cosine_maxabs(fwd_attn_out, exp_attn_out, S*D, &mf));
    printf("FWD res1     : cos=%.6f\n", cosine_maxabs(fwd_res1, exp_res1, S*D, &mf));
    printf("FWD n2       : cos=%.6f\n", cosine_maxabs(fwd_n2, exp_n2, S*D, &mf));
    printf("FWD res2     : cos=%.6f\n", cosine_maxabs(fwd_res2, exp_res2, S*D, &mf));

    // seed grad
    bmt_scheduler_set_grad(sched, t_res2, gout, S*D*sizeof(float));
    bmt_scheduler_backward(sched);

    float *ginp=malloc(S*D*4), *grmsw1=malloc(D*4), *gwq=malloc(D*D*4), *gwk=malloc(KV*D*4), *gwv=malloc(KV*D*4),
          *gwproj=malloc(D*D*4), *grmsw2=malloc(D*4), *gwgate=malloc(HID*D*4), *gwup=malloc(HID*D*4), *gwdown=malloc(D*HID*4);
    
    bmt_scheduler_get_grad(sched, t_inp, ginp, S*D*4);
    bmt_scheduler_get_grad(sched, t_rmsw1, grmsw1, D*4);
    bmt_scheduler_get_grad(sched, t_wq, gwq, D*D*4);
    bmt_scheduler_get_grad(sched, t_wk, gwk, KV*D*4);
    bmt_scheduler_get_grad(sched, t_wv, gwv, KV*D*4);
    bmt_scheduler_get_grad(sched, t_wproj, gwproj, D*D*4);
    bmt_scheduler_get_grad(sched, t_rmsw2, grmsw2, D*4);
    bmt_scheduler_get_grad(sched, t_wgate, gwgate, HID*D*4);
    bmt_scheduler_get_grad(sched, t_wup, gwup, HID*D*4);
    bmt_scheduler_get_grad(sched, t_wdown, gwdown, D*HID*4);

    float m1,m2,m3,m4,m5,m6,m7,m8,m9,m10;
    double c1=cosine_maxabs(ginp, exp_ginp, S*D, &m1);
    double c2=cosine_maxabs(grmsw1, exp_grmsw1, D, &m2);
    double c3=cosine_maxabs(gwq, exp_gwq, D*D, &m3);
    double c4=cosine_maxabs(gwk, exp_gwk, KV*D, &m4);
    double c5=cosine_maxabs(gwv, exp_gwv, KV*D, &m5);
    double c6=cosine_maxabs(gwproj, exp_gwproj, D*D, &m6);
    double c7=cosine_maxabs(grmsw2, exp_grmsw2, D, &m7);
    double c8=cosine_maxabs(gwgate, exp_gwgate, HID*D, &m8);
    double c9=cosine_maxabs(gwup, exp_gwup, HID*D, &m9);
    double c10=cosine_maxabs(gwdown, exp_gwdown, D*HID, &m10);

    printf("grad_inp   : cos=%.6f maxabs=%.2e\n", c1, m1);
    printf("grad_rmsw1 : cos=%.6f maxabs=%.2e\n", c2, m2);
    printf("grad_wq    : cos=%.6f maxabs=%.2e\n", c3, m3);
    printf("grad_wk    : cos=%.6f maxabs=%.2e\n", c4, m4);
    printf("grad_wv    : cos=%.6f maxabs=%.2e\n", c5, m5);
    printf("grad_wproj : cos=%.6f maxabs=%.2e\n", c6, m6);
    printf("grad_rmsw2 : cos=%.6f maxabs=%.2e\n", c7, m7);
    printf("grad_wgate : cos=%.6f maxabs=%.2e\n", c8, m8);
    printf("grad_wup   : cos=%.6f maxabs=%.2e\n", c9, m9);
    printf("grad_wdown : cos=%.6f maxabs=%.2e\n", c10, m10);

    int ok = (c1>0.999)&&(c2>0.999)&&(c3>0.999)&&(c4>0.999)&&(c5>0.999)&&(c6>0.999)&&(c7>0.999)&&(c8>0.999)&&(c9>0.999)&&(c10>0.999);
    printf("\n=== full layer backward G2 test: %s ===\n", ok?"PASS":"FAIL");

    bmt_scheduler_destroy(sched); bmk_registry_destroy(reg); bmt_graph_destroy(g);
    backend_destroy(be);
    return ok?0:1;
}
