/* fwd_train_test.c -- Step 2: full-model forward_train validation.
 * Loads real SmolLM2-135M, builds the TRAINING graph (bmt_graph_build_train),
 * does embedding lookup, forward_train, and compares logits [S,V] to PyTorch.
 * Forward cos=1.0 = the multi-layer training forward works end-to-end. */
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
    const char* md = "data/smollm2-135m";
    const int tokens[] = {6403, 1980, 253, 655};   /* "Once upon a time" */
    const int S = 4;
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }

    bm_model_t* m = calloc(1, sizeof(bm_model_t));
    bm_load_weights(m, md);
    int D = m->arch.dim, V = m->arch.vocab_size, L = m->arch.n_layers;
    fprintf(stderr, "[fwd_train] dim=%d V=%d L=%d kv=%d\n", D, V, L, m->kv_dim);

    bmt_graph_build_train(m, S);
    bmt_graph_t* g = (bmt_graph_t*)m->graph;
    fprintf(stderr, "[fwd_train] graph: %d tensors, %d nodes\n", g->n_tensors, g->n_nodes);

    bmk_registry_t* reg = bmk_registry_create(be);
    bmk_register(reg, BMK_OP_MATMUL, BMK_VARIANT_NAIVE, "matmul_forward_naive");
    bmk_register(reg, BMK_OP_NORM_RMS, BMK_VARIANT_NAIVE, "rmsnorm_forward");
    bmk_register(reg, BMK_OP_ACT_SWIGLU, BMK_VARIANT_NAIVE, "swiglu_forward");
    bmt_scheduler_t* sched = bmt_scheduler_create(be, reg, g, S, m->kv_dim, L);

    /* embedding lookup: x[s,d] = wte[token[s]*D + d]  (SmolLM2: no scale, no wpe) */
    float* x = malloc((size_t)S*D*sizeof(float));
    for (int s = 0; s < S; s++)
        memcpy(x + s*D, m->token_embedding_table + tokens[s]*D, D*sizeof(float));
    bmt_scheduler_set_input(sched, 0, x, (size_t)S*D*sizeof(float));

    bmt_scheduler_forward_train(sched, S);

    int t_logits = g->n_tensors - 1;
    float* logits = malloc((size_t)S*V*sizeof(float));
    bmt_scheduler_get_output(sched, t_logits, logits, (size_t)S*V*sizeof(float));

    float* ref = loadf("test/grad", "fwd_train_logits.bin", (size_t)S*V);
    int fails = 0;
    for (int s = 0; s < S; s++) {
        float mx; double c = cosine_maxabs(logits + s*V, ref + s*V, V, &mx);
        /* top-1 check */
        int our_top=0, ref_top=0;
        for (int i=1;i<V;i++){ if(logits[s*V+i]>logits[s*V+our_top]) our_top=i; if(ref[s*V+i]>ref[s*V+ref_top]) ref_top=i; }
        printf("pos %d: cos=%.6f maxabs=%.2e top_ours=%d top_ref=%d %s\n",
               s, c, mx, our_top, ref_top, (c>0.9999 && our_top==ref_top)?"OK":"MISMATCH");
        if (c<=0.9999 || our_top!=ref_top) fails++;
    }
    int ok = (fails==0);
    printf("\n=== full-model forward_train (SmolLM2-135M, S=%d): %s ===\n", S, ok?"PASS":"FAIL");

    free(x); free(logits); free(ref);
    bmt_scheduler_destroy(sched); bmk_registry_destroy(reg);
    bm_destroy_model(m); backend_destroy(be);
    return ok?0:1;
}
