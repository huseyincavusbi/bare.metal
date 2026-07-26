#ifndef BMT_GRAPH_H
#define BMT_GRAPH_H

#include "kernels/registry.h"
#include <stddef.h>
#include <stdint.h>

#define BMT_MAX_TENSOR_DIMS 4
#define BMT_MAX_NODE_INPUTS 8
#define BMT_MAX_NODE_PARAMS 8

typedef enum {
    BMT_TENSOR_TYPE_ACTIVATION = 0,
    BMT_TENSOR_TYPE_WEIGHT,
    BMT_TENSOR_TYPE_IO
} bmt_tensor_type_t;

typedef struct {
    int id;
    bmt_tensor_type_t type;
    int dims[BMT_MAX_TENSOR_DIMS];
    int n_dims;
    void* weight_ptr;
    int   quantized;   /* 1 = upload as Q8 blocks, dispatch via matmul_forward_q8 */
} bmt_tensor_t;

typedef struct {
    int id;
    bmk_op_type_t op_type;
    int inputs[BMT_MAX_NODE_INPUTS];
    int n_inputs;
    int output;
    int params[BMT_MAX_NODE_PARAMS];
    int n_params;
    float fparams[BMT_MAX_NODE_PARAMS];
    int n_fparams;
    bmk_variant_t variant;
} bmt_node_t;

typedef struct bmt_graph_s {
    bmt_tensor_t* tensors;
    int n_tensors;
    int capacity_tensors;

    bmt_node_t* nodes;
    int n_nodes;
    int capacity_nodes;
} bmt_graph_t;

bmt_graph_t* bmt_graph_create(void);
void bmt_graph_destroy(bmt_graph_t* graph);

int bmt_graph_add_tensor(bmt_graph_t* graph, bmt_tensor_type_t type, int n_dims, const int* dims);
void bmt_graph_set_weight_ptr(bmt_graph_t* graph, int tensor_id, void* ptr);

int bmt_graph_add_node(bmt_graph_t* graph, bmk_op_type_t op_type, 
                       int n_inputs, const int* inputs, int output, 
                       int n_params, const int* params,
                       int n_fparams, const float* fparams);

// High-level API to build the graph for a model
struct bm_model_s;
void bmt_graph_build(struct bm_model_s* model);

#endif
