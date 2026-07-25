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

int bmt_graph_add_weight(bmt_graph_t* graph, void* ptr, int n_dims, const int* dims) {
    int id = bmt_graph_add_tensor(graph, BMT_TENSOR_TYPE_WEIGHT, n_dims, dims);
    bmt_graph_set_weight_ptr(graph, id, ptr);
    return id;
}

void bmt_graph_build(bm_model_t* m) {
    if (m->graph) bmt_graph_destroy(m->graph);
    bmt_graph_t* g = bmt_graph_create();
    m->graph = g;

    int D = m->arch.dim;
    int H = m->arch.hidden_dim;
    int NH = m->arch.n_heads;
    int HD = m->head_size;
    int KV = m->kv_dim;
    int NKV = m->n_kv_heads;
    int V = m->arch.vocab_size;
    int L = m->arch.n_layers;

    // The residual stream
    int t_x = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});

    for (int l = 0; l < L; l++) {
        // Norm 1
        int t_ln1w = bmt_graph_add_weight(g, m->ln1w + l*D, 1, (int[]){D});
        int t_ln1b = m->ln1b ? bmt_graph_add_weight(g, m->ln1b + l*D, 1, (int[]){D}) : -1;
        int t_norm1 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});
        
        bmk_op_type_t norm_op = m->arch.norm == BM_NORM_RMSNORM ? BMK_OP_NORM_RMS : BMK_OP_NORM_LAYER;
        if (m->arch.norm == BM_NORM_RMSNORM) {
            bmt_graph_add_node(g, norm_op, 2, (int[]){t_x, t_ln1w}, t_norm1, 2, (int[]){1, D}, 1, (float[]){1e-5f});
        } else {
            bmt_graph_add_node(g, norm_op, 3, (int[]){t_x, t_ln1w, t_ln1b}, t_norm1, 2, (int[]){1, D}, 1, (float[]){1e-5f});
        }

        // Q, K, V Matmuls
        int t_qw = bmt_graph_add_weight(g, m->qw + l*NH*HD*D, 2, (int[]){NH*HD, D});
        int t_kw = bmt_graph_add_weight(g, m->kw + l*NKV*HD*D, 2, (int[]){KV, D});
        int t_vw = bmt_graph_add_weight(g, m->vw + l*NKV*HD*D, 2, (int[]){KV, D});
        
        int t_q = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){NH*HD});
        int t_k = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){KV});
        int t_v = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){KV});

        bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_norm1, t_qw}, t_q, 3, (int[]){1, D, NH*HD}, 0, NULL);
        bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_norm1, t_kw}, t_k, 3, (int[]){1, D, KV}, 0, NULL);
        bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_norm1, t_vw}, t_v, 3, (int[]){1, D, KV}, 0, NULL);

        // RoPE
        if (m->arch.pos_enc == BM_POS_ROPE) {
            bmt_graph_add_node(g, BMK_OP_POS_ENC_ROPE, 2, (int[]){t_q, t_k}, t_q, 3, (int[]){HD, m->n_kv_heads, NH}, 1, (float[]){m->arch.rope_theta});
        }

        // Attention
        int t_attn_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){NH*HD});
        bmt_graph_add_node(g, BMK_OP_ATTENTION, 3, (int[]){t_q, t_k, t_v}, t_attn_out, 8, (int[]){NH, HD, NKV, KV, 1, m->kv_mul, l, m->arch.max_seq_len}, 0, NULL);

        // Attn Proj
        int t_attprojw = bmt_graph_add_weight(g, m->attprojw + l*NH*HD*D, 2, (int[]){D, NH*HD});
        int t_proj_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});
        bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_attn_out, t_attprojw}, t_proj_out, 3, (int[]){1, NH*HD, D}, 0, NULL);

        // Residual Add 1 (in-place on t_x)
        bmt_graph_add_node(g, BMK_OP_ADD, 2, (int[]){t_x, t_proj_out}, t_x, 1, (int[]){D}, 0, NULL);
        
        // Norm 2
        int t_ln2 = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});
        int t_ln2w = bmt_graph_add_weight(g, m->ln2w + l*D, 1, (int[]){D});
        if (m->arch.norm == BM_NORM_RMSNORM) {
            bmt_graph_add_node(g, BMK_OP_NORM_RMS, 2, (int[]){t_x, t_ln2w}, t_ln2, 2, (int[]){1, D}, 1, (float[]){1e-5f});
        } else {
            int t_ln2b = m->ln2b ? bmt_graph_add_weight(g, m->ln2b + l*D, 1, (int[]){D}) : -1;
            bmt_graph_add_node(g, BMK_OP_NORM_LAYER, 3, (int[]){t_x, t_ln2w, t_ln2b}, t_ln2, 2, (int[]){1, D}, 1, (float[]){1e-5f});
        }
        
        // FFN
        int t_fcw_out;
        if (m->arch.gated_mlp) {
            int t_fcw = bmt_graph_add_weight(g, m->fcw + l*H*D, 2, (int[]){D, H});
            int t_fcw3 = bmt_graph_add_weight(g, m->fcw3 + l*H*D, 2, (int[]){D, H});
            t_fcw_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){H});
            int t_fcw3_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){H});
            
            bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_ln2, t_fcw}, t_fcw_out, 3, (int[]){1, D, H}, 0, NULL);
            bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_ln2, t_fcw3}, t_fcw3_out, 3, (int[]){1, D, H}, 0, NULL);
            
            if (m->arch.activation == BM_ACT_SWIGLU) {
                bmt_graph_add_node(g, BMK_OP_ACT_SWIGLU, 2, (int[]){t_fcw_out, t_fcw3_out}, t_fcw_out, 1, (int[]){H}, 0, NULL);
            } else {
                bmt_graph_add_node(g, BMK_OP_ACT_GELU, 1, (int[]){t_fcw_out}, t_fcw_out, 1, (int[]){H}, 0, NULL);
            }
        } else {
            int t_fcw = bmt_graph_add_weight(g, m->fcw + l*H*D, 2, (int[]){D, H});
            t_fcw_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){H});
            bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_ln2, t_fcw}, t_fcw_out, 3, (int[]){1, D, H}, 0, NULL);
            bmt_graph_add_node(g, BMK_OP_ACT_GELU, 1, (int[]){t_fcw_out}, t_fcw_out, 1, (int[]){H}, 0, NULL);
        }
        
        int t_fcprojw = bmt_graph_add_weight(g, m->fcprojw + l*D*H, 2, (int[]){H, D});
        int t_ffn_out = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});
        bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_fcw_out, t_fcprojw}, t_ffn_out, 3, (int[]){1, H, D}, 0, NULL);
        
        // Residual Add 2 (in-place on t_x)
        bmt_graph_add_node(g, BMK_OP_ADD, 2, (int[]){t_x, t_ffn_out}, t_x, 1, (int[]){D}, 0, NULL);
    }

    // Final Classifier
    int t_lnfw = bmt_graph_add_weight(g, m->lnfw, 1, (int[]){D});
    int t_lnfb = m->lnfb ? bmt_graph_add_weight(g, m->lnfb, 1, (int[]){D}) : -1;
    int t_norm_f = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){D});
    
    bmk_op_type_t norm_op = m->arch.norm == BM_NORM_RMSNORM ? BMK_OP_NORM_RMS : BMK_OP_NORM_LAYER;
    if (m->arch.norm == BM_NORM_RMSNORM) {
        bmt_graph_add_node(g, norm_op, 2, (int[]){t_x, t_lnfw}, t_norm_f, 2, (int[]){1, D}, 1, (float[]){1e-5f});
    } else {
        bmt_graph_add_node(g, norm_op, 3, (int[]){t_x, t_lnfw, t_lnfb}, t_norm_f, 2, (int[]){1, D}, 1, (float[]){1e-5f});
    }

    int t_wcls = bmt_graph_add_weight(g, m->wcls, 2, (int[]){V, D});
    int t_logits = bmt_graph_add_tensor(g, BMT_TENSOR_TYPE_ACTIVATION, 1, (int[]){V});
    bmt_graph_add_node(g, BMK_OP_MATMUL, 2, (int[]){t_norm_f, t_wcls}, t_logits, 3, (int[]){1, D, V}, 0, NULL);
}
