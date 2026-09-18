/* graph_capacity_test.c -- regression test for graph growth initialization.
 *
 * The graph's tensor/node arrays start at capacity 128 (calloc'd) and grow with
 * realloc, which does NOT zero new memory. A missing field initialization in
 * bmt_graph_add_tensor (notably `quantized`) caused garbage values for every
 * tensor >= 128, which in turn made the scheduler mis-allocate weight buffers
 * (Q8-sized buffers written as bf16) and produced silent corruption / NaN /
 * SIGBUS.
 *
 * To catch this deterministically (independent of heap contents) the test
 * poisons every allocated slot before adding, so any field that
 * bmt_graph_add_tensor/​add_node fails to initialize is exposed. It then grows
 * the graph well past the initial capacity and checks every entry. */
#include "baremetal/graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    bmt_graph_t* g = bmt_graph_create();
    if (!g) { printf("FAIL: graph create\n"); return 1; }

    /* Poison every allocated slot so uninitialized fields are non-zero. */
    for (int i = 0; i < g->capacity_tensors; i++)
        memset(&g->tensors[i], 0xAB, sizeof(bmt_tensor_t));
    for (int i = 0; i < g->capacity_nodes; i++)
        memset(&g->nodes[i], 0xAB, sizeof(bmt_node_t));

    const int N = 400;                 /* >> initial capacity (128) */
    const int dims[2] = { 4, 8 };
    const int in[2] = { 0, 1 };
    const int pr[2] = { 1, 2 };
    int bad = 0;

    for (int i = 0; i < N; i++) {
        int id = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 2, dims);
        if (id != i) { printf("FAIL: tensor id %d != %d\n", id, i); return 1; }
        bmt_tensor_t* t = &g->tensors[id];
        if (t->id != i || t->type != BMT_TENSOR_TYPE_ACTIVATION ||
            t->n_dims != 2 || t->dims[0] != dims[0] || t->dims[1] != dims[1] ||
            t->weight_ptr != NULL || t->quantized != 0) {
            printf("FAIL: tensor %d not cleanly initialized "
                   "(id=%d type=%d ndims=%d dims=%d,%d wp=%p quant=%d)\n",
                   i, t->id, (int)t->type, t->n_dims, t->dims[0], t->dims[1],
                   t->weight_ptr, t->quantized);
            if (++bad > 8) break;
        }

        int nid = bmt_graph_add_node(g, BMK_OP_MATMUL, 2, in, 0, 2, pr, 0, NULL);
        bmt_node_t* n = &g->nodes[nid];
        if (n->id != nid || n->op_type != BMK_OP_MATMUL || n->n_inputs != 2 ||
            n->output != 0 || n->n_params != 2 || n->n_fparams != 0 ||
            n->variant != BMK_VARIANT_NAIVE) {
            printf("FAIL: node %d not cleanly initialized "
                   "(id=%d op=%d nin=%d out=%d npar=%d nfp=%d var=%d)\n",
                   nid, n->id, (int)n->op_type, n->n_inputs, n->output,
                   n->n_params, n->n_fparams, (int)n->variant);
            if (++bad > 8) break;
        }
    }

    bmt_graph_destroy(g);

    if (bad) { printf("=== graph_capacity_test: FAIL (%d)\n", bad); return 1; }
    printf("=== graph_capacity_test: PASS (tensors=%d nodes=%d) ===\n", N, N);
    return 0;
}
