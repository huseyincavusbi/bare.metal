#include "baremetal/graph.h"
#include "baremetal/model.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>

bmt_graph_t* bmt_graph_create(void) {
    bmt_graph_t* g = calloc(1, sizeof(bmt_graph_t));
    if (!g) return NULL;
    g->capacity_tensors = 128;
    g->tensors = calloc(g->capacity_tensors, sizeof(bmt_tensor_t));
    g->capacity_nodes = 128;
    g->nodes = calloc(g->capacity_nodes, sizeof(bmt_node_t));
    return g;
}

void bmt_graph_destroy(bmt_graph_t* g) {
    if (!g) return;
    free(g->tensors);
    free(g->nodes);
    free(g);
}

int bmt_graph_add_tensor(bmt_graph_t* graph, bmt_tensor_type_t type, int n_dims, const int* dims) {
    if (graph->n_tensors >= graph->capacity_tensors) {
        graph->capacity_tensors *= 2;
        graph->tensors = realloc(graph->tensors, graph->capacity_tensors * sizeof(bmt_tensor_t));
    }
    int id = graph->n_tensors++;
    bmt_tensor_t* t = &graph->tensors[id];
    t->id = id;
    t->type = type;
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims && i < BMT_MAX_TENSOR_DIMS; i++) t->dims[i] = dims[i];
    t->weight_ptr = NULL;
    return id;
}

void bmt_graph_set_weight_ptr(bmt_graph_t* graph, int tensor_id, void* ptr) {
    if (tensor_id >= 0 && tensor_id < graph->n_tensors) {
        graph->tensors[tensor_id].weight_ptr = ptr;
    }
}

int bmt_graph_add_node(bmt_graph_t* graph, bmk_op_type_t op_type, 
                       int n_inputs, const int* inputs, int output, 
                       int n_params, const int* params,
                       int n_fparams, const float* fparams) {
    if (graph->n_nodes >= graph->capacity_nodes) {
        graph->capacity_nodes *= 2;
        graph->nodes = realloc(graph->nodes, graph->capacity_nodes * sizeof(bmt_node_t));
    }
    int id = graph->n_nodes++;
    bmt_node_t* n = &graph->nodes[id];
    n->id = id;
    n->op_type = op_type;
    n->n_inputs = n_inputs;
    for (int i = 0; i < n_inputs && i < BMT_MAX_NODE_INPUTS; i++) n->inputs[i] = inputs[i];
    n->output = output;
    n->n_params = n_params;
    for (int i = 0; i < n_params && i < BMT_MAX_NODE_PARAMS; i++) n->params[i] = params[i];
    n->n_fparams = n_fparams;
    for (int i = 0; i < n_fparams && i < BMT_MAX_NODE_PARAMS; i++) n->fparams[i] = fparams[i];
    n->variant = BMK_VARIANT_NAIVE;
    return id;
}

void bmt_graph_build(bm_model_t* model) {
    if (model->graph) bmt_graph_destroy(model->graph);
    bmt_graph_t* g = bmt_graph_create();
    model->graph = g;

    // TODO: Build the complete computational graph based on model->arch
    // We will populate this with the same sequence of operations currently found in tools/run.c
}
