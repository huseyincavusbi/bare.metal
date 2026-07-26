/* bwd_full_test.c -- Step 3: full-model backward (G2-full).
 * SmolLM2-135M: forward_train -> xent_backward (seed) -> scheduler backward walk,
 * then compare a representative sample of weight grads (layers 0/15/29 + lnfw)
 * to PyTorch autograd. cos=1.0 = the full 30-layer backward chain is correct. */
#include "baremetal.h"
#include "baremetal/model.h"
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
    char path[256]; snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
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
static int find_w(bmt_graph_t* g, void* ptr) {
    for (int i=0;i<g->n_tensors;i++)
        if (g->tensors[i].type==BMT_TENSOR_TYPE_WEIGHT && g->tensors[i].weight_ptr==ptr) return i;
    return -1;
}

int main(void) {
    const char* md = "data/smollm2-135m";
    const int tokens[] = {6403, 1980, 253, 655};
    const int targets_arr[] = {1980, 253, 655, 0};
    const int S = 4;
    const char* G = "test/grad";
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, md);
    int D=m->arch.dim, V=m->arch.vocab_size, L=m->arch.n_layers;
    int NH=m->arch.n_heads, HD=m->head_size, NKV=m->n_kv_heads, H=m->arch.hidden_dim;
    bmt_graph_build_train(m, S);
    bmt_graph_t* g = (bmt_graph_t*)m->graph;

    bmk_registry_t* reg = bmk_registry_create(be);
    bmk_register(reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");
    bmt_scheduler_t* sched = bmt_scheduler_create(be, reg, g, S, m->kv_dim, L);

    float* x = malloc((size_t)S*D*sizeof(float));
    for (int s=0;s<S;s++) memcpy(x+s*D, m->token_embedding_table+tokens[s]*D, D*sizeof(float));
    bmt_scheduler_set_input(sched, 0, x, (size_t)S*D*sizeof(float));
    bmt_scheduler_forward_train(sched, S);

    int t_logits = g->n_tensors - 1;

    float loss = bmt_scheduler_xent_backward(sched, t_logits, targets_arr, S, V);
    printf("[G2-full] loss=%.4f\n", loss);

    bmt_scheduler_backward(sched);

    int fails = 0;
    struct { const char* nm; void* ptr; size_t n; } sm[] = {
        {"bwd_l0_qw",  m->qw + 0*NH*HD*D,        (size_t)NH*HD*D},
        {"bwd_l0_kw",  m->kw + 0*NKV*HD*D,       (size_t)NKV*HD*D},
        {"bwd_l0_vw",  m->vw + 0*NKV*HD*D,       (size_t)NKV*HD*D},
        {"bwd_l0_ow",  m->attprojw + 0*NH*HD*D,  (size_t)NH*HD*D},
        {"bwd_l0_ln1w",m->ln1w + 0*D,            (size_t)D},
        {"bwd_l0_ln2w",m->ln2w + 0*D,            (size_t)D},
        {"bwd_l0_gw",  m->fcw + 0*H*D,           (size_t)H*D},
        {"bwd_l0_uw",  m->fcw3 + 0*H*D,          (size_t)H*D},
        {"bwd_l0_dw",  m->fcprojw + 0*D*H,       (size_t)D*H},
        {"bwd_l15_qw", m->qw + 15*NH*HD*D,       (size_t)NH*HD*D},
        {"bwd_l15_ow", m->attprojw + 15*NH*HD*D, (size_t)NH*HD*D},
        {"bwd_l15_dw", m->fcprojw + 15*D*H,      (size_t)D*H},
        {"bwd_l29_qw", m->qw + 29*NH*HD*D,       (size_t)NH*HD*D},
        {"bwd_l29_ow", m->attprojw + 29*NH*HD*D, (size_t)NH*HD*D},
        {"bwd_l29_dw", m->fcprojw + 29*D*H,      (size_t)D*H},
        {"bwd_lnfw",   m->lnfw,                  (size_t)D},
    };
    int ns = sizeof(sm)/sizeof(sm[0]);
    for (int i=0;i<ns;i++) {
        int tid = find_w(g, sm[i].ptr);
        if (tid < 0) { printf("FAIL %-12s: tensor not found\n", sm[i].nm); fails++; continue; }
        float* our = malloc(sm[i].n*sizeof(float));
        bmt_scheduler_get_grad(sched, tid, our, sm[i].n*sizeof(float));
        float* ref = loadf(G, sm[i].nm, sm[i].n);
        float mx; double c = cosine_maxabs(our, ref, sm[i].n, &mx);
        printf("%-12s cos=%.6f maxabs=%.2e %s\n", sm[i].nm, c, mx, c>0.9999?"OK":"MISMATCH");
        if (c<=0.9999) fails++;
        free(our); free(ref);
    }
    int ok = (fails==0);
    printf("\n=== full-model backward G2 (SmolLM2-135M, 30 layers): %s (%d/%d ok) ===\n",
           ok?"PASS":"FAIL", ns-fails, ns);

    free(x);
    bmt_scheduler_destroy(sched); bmk_registry_destroy(reg);
    bm_destroy_model(m); backend_destroy(be);
    return ok?0:1;
}
