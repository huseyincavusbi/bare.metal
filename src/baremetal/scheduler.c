#include "baremetal/scheduler.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>

struct bmt_scheduler_s {
    backend_ctx_t* backend;
    bmk_registry_t* reg;
    bmt_graph_t* graph;
    
    backend_buffer_t** buffers;
    
    backend_buffer_t* param_buf;
    backend_buffer_t* eps_buf;
};

bmt_scheduler_t* bmt_scheduler_create(backend_ctx_t* backend, bmk_registry_t* reg, bmt_graph_t* graph) {
    bmt_scheduler_t* sched = calloc(1, sizeof(bmt_scheduler_t));
    sched->backend = backend;
    sched->reg = reg;
    sched->graph = graph;
    
    sched->buffers = calloc(graph->n_tensors, sizeof(backend_buffer_t*));
    for (int i = 0; i < graph->n_tensors; i++) {
        bmt_tensor_t* t = &graph->tensors[i];
        
        size_t size = 1;
        for (int d = 0; d < t->n_dims; d++) size *= t->dims[d];
        if (size == 0) size = 1024;
        size *= sizeof(float);
        
        sched->buffers[i] = backend_buffer_alloc(backend, size);
        
        if (t->type == BMT_TENSOR_TYPE_WEIGHT && t->weight_ptr) {
            void* dst = backend_buffer_map(sched->buffers[i]);
            memcpy(dst, t->weight_ptr, size);
            backend_buffer_unmap(sched->buffers[i]);
        }
    }
    
    sched->param_buf = backend_buffer_alloc(backend, 16 * sizeof(int));
    sched->eps_buf = backend_buffer_alloc(backend, sizeof(float));
    
    void* eps_ptr = backend_buffer_map(sched->eps_buf);
    *(float*)eps_ptr = 1e-5f;
    backend_buffer_unmap(sched->eps_buf);
    
    return sched;
}

void bmt_scheduler_destroy(bmt_scheduler_t* sched) {
    if (!sched) return;
    for (int i = 0; i < sched->graph->n_tensors; i++) {
        if (sched->buffers[i]) backend_buffer_free(sched->buffers[i]);
    }
    free(sched->buffers);
    backend_buffer_free(sched->param_buf);
    backend_buffer_free(sched->eps_buf);
    free(sched);
}

void bmt_scheduler_run(bmt_scheduler_t* sched, int pos, int seq_len) {
    (void)pos; (void)seq_len;
    backend_encoder_t* enc = backend_encode_begin(sched->backend);
    
    for (int i = 0; i < sched->graph->n_nodes; i++) {
        bmt_node_t* node = &sched->graph->nodes[i];
        if (node->op_type == BMK_OP_COUNT) continue;
        
        int p1 = 1, p2 = 1, p3 = 1;
        if (node->op_type == BMK_OP_MATMUL) {
            p1 = node->params[0]; p2 = node->params[1]; p3 = node->params[2];
        } else if (node->op_type == BMK_OP_NORM_RMS || node->op_type == BMK_OP_FUSED_RESIDUAL_NORM) {
            p1 = node->params[0]; p2 = node->params[1]; p3 = node->params[1];
        } else if (node->op_type == BMK_OP_FUSED_CLASSIFIER) {
            p1 = 1; p2 = node->params[0]; p3 = node->params[1];
        } else if (node->op_type == BMK_OP_ACT_SWIGLU || node->op_type == BMK_OP_ACT_GELU) {
            p1 = 1; p2 = node->params[0]; p3 = node->params[0];
        } else if (node->op_type == BMK_OP_POS_ENC_ROPE) {
            p1 = 1; p2 = node->params[0]; p3 = node->params[0];
        }
        
        backend_kernel_t* kn = bmk_select(sched->reg, node->op_type, p1, p2, p3);
        if (!kn) continue;
        
        backend_buffer_t* bufs[12];
        int n_bufs = 0;
        
        for (int j = 0; j < node->n_inputs; j++) {
            bufs[n_bufs++] = sched->buffers[node->inputs[j]];
        }
        bufs[n_bufs++] = sched->buffers[node->output];
        
        int* p = backend_buffer_map(sched->param_buf);
        for(int j=0; j<node->n_params; j++) p[j] = node->params[j];
        backend_buffer_unmap(sched->param_buf);
        
        bufs[n_bufs++] = sched->param_buf;
        
        if (node->n_fparams > 0) {
            bufs[n_bufs++] = sched->eps_buf;
        }
        
        int gtx = 1, gty = 1, gtz = 1;
        int ttx = 32, tty = 1, ttz = 1;
        
        if (node->op_type == BMK_OP_MATMUL) {
            gtx = (p3 + 31) & ~31;
            ttx = 32;
        } else if (node->op_type == BMK_OP_NORM_RMS || node->op_type == BMK_OP_FUSED_RESIDUAL_NORM) {
            gtx = 256;
            ttx = 256;
        } else if (node->op_type == BMK_OP_FUSED_CLASSIFIER) {
            gtx = (p3 + 255) & ~255;
            ttx = 256;
        } else if (node->op_type == BMK_OP_ACT_SWIGLU || node->op_type == BMK_OP_ACT_GELU) {
            gtx = p2;
            ttx = p2 < 256 ? p2 : 256;
        } else if (node->op_type == BMK_OP_ADD) {
            gtx = (p2 + 255) & ~255;
            ttx = 256;
        }
        
        backend_encode_dispatch(enc, kn, bufs, NULL, n_bufs, gtx, gty, gtz, ttx, tty, ttz);
    }
    
    backend_encode_commit(enc);
    backend_encode_wait(enc);
}
