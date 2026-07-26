#include "baremetal/scheduler.h"
#include "baremetal/quant.h"
#include "utils/log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct bmt_scheduler_s {
    backend_ctx_t* backend;
    bmk_registry_t* reg;
    bmt_graph_t* graph;
    
    backend_buffer_t** buffers;
    
    backend_buffer_t** param_bufs;
    backend_buffer_t** eps_bufs;
    backend_buffer_t** pos_bufs;
    backend_buffer_t** nkv_bufs;
    
    backend_buffer_t* bkvc;
    backend_buffer_t* dummy_bias;
    backend_kernel_t* q8_kernel;   /* matmul_forward_q8, used for quantized weights */
    
    int max_seq_len;
    int kv_dim;
    int n_layers;
};

bmt_scheduler_t* bmt_scheduler_create(backend_ctx_t* backend, bmk_registry_t* reg, bmt_graph_t* graph, int max_seq_len, int kv_dim, int n_layers) {
    bmt_scheduler_t* sched = calloc(1, sizeof(bmt_scheduler_t));
    sched->backend = backend;
    sched->reg = reg;
    sched->graph = graph;
    
    sched->buffers = calloc(graph->n_tensors, sizeof(backend_buffer_t*));
    for (int i = 0; i < graph->n_tensors; i++) {
        bmt_tensor_t* t = &graph->tensors[i];
        
        size_t n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        if (n_elems == 0) n_elems = 1024;
        size_t fp32_bytes = n_elems * sizeof(float);
        
        int do_q8 = (t->type == BMT_TENSOR_TYPE_WEIGHT && t->weight_ptr && t->quantized
                     && t->n_dims == 2 && (t->dims[1] % 32 == 0));
        
        if (do_q8) {
            size_t qbytes = bmt_q8_bytes(n_elems);
            sched->buffers[i] = backend_buffer_alloc(backend, qbytes);
            q8_block_t* dst = (q8_block_t*)backend_buffer_map(sched->buffers[i]);
            bmt_quantize_q8((const float*)t->weight_ptr, dst, n_elems);
            backend_buffer_unmap(sched->buffers[i]);
        } else {
            sched->buffers[i] = backend_buffer_alloc(backend, fp32_bytes);
            if (t->type == BMT_TENSOR_TYPE_WEIGHT && t->weight_ptr) {
                void* dst = backend_buffer_map(sched->buffers[i]);
                memcpy(dst, t->weight_ptr, fp32_bytes);
                backend_buffer_unmap(sched->buffers[i]);
            }
        }
    }
    
    sched->param_bufs = malloc(graph->n_nodes * sizeof(backend_buffer_t*));
    sched->eps_bufs = malloc(graph->n_nodes * sizeof(backend_buffer_t*));
    sched->pos_bufs = malloc(graph->n_nodes * sizeof(backend_buffer_t*));
    sched->nkv_bufs = malloc(graph->n_nodes * sizeof(backend_buffer_t*));
    
    for (int i = 0; i < graph->n_nodes; i++) {
        sched->param_bufs[i] = backend_buffer_alloc(backend, 16 * sizeof(int));
        sched->eps_bufs[i] = backend_buffer_alloc(backend, 16 * sizeof(float));
        sched->pos_bufs[i] = backend_buffer_alloc(backend, sizeof(int));
        sched->nkv_bufs[i] = backend_buffer_alloc(backend, sizeof(int));
    }
    
    sched->max_seq_len = max_seq_len;
    sched->kv_dim = kv_dim;
    sched->n_layers = n_layers;
    sched->bkvc = backend_buffer_alloc(backend, n_layers * 2 * max_seq_len * kv_dim * sizeof(float));
    
    sched->dummy_bias = backend_buffer_alloc(backend, 131072 * sizeof(float));
    void* db = backend_buffer_map(sched->dummy_bias);
    memset(db, 0, 131072 * sizeof(float));
    backend_buffer_unmap(sched->dummy_bias);
    
    sched->q8_kernel = backend_kernel_create(backend, "matmul_forward_q8");
    
    return sched;
}

void bmt_scheduler_destroy(bmt_scheduler_t* sched) {
    if (!sched) return;
    for (int i = 0; i < sched->graph->n_tensors; i++) {
        if (sched->buffers[i]) backend_buffer_free(sched->buffers[i]);
    }
    free(sched->buffers);
    for (int i = 0; i < sched->graph->n_nodes; i++) {
        backend_buffer_free(sched->param_bufs[i]);
        backend_buffer_free(sched->eps_bufs[i]);
        backend_buffer_free(sched->pos_bufs[i]);
        backend_buffer_free(sched->nkv_bufs[i]);
    }
    free(sched->param_bufs);
    free(sched->eps_bufs);
    free(sched->pos_bufs);
    free(sched->nkv_bufs);
    backend_buffer_free(sched->bkvc);
    backend_buffer_free(sched->dummy_bias);
    if (sched->q8_kernel) backend_kernel_destroy(sched->q8_kernel);
    free(sched);
}

void bmt_scheduler_run(bmt_scheduler_t* sched, int pos, int seq_len) {
    (void)seq_len;
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
            p1 = 1; p2 = node->params[1]; p3 = node->params[2]; // C=D, V (grid runs over V outputs)
        } else if (node->op_type == BMK_OP_ACT_SWIGLU || node->op_type == BMK_OP_ACT_GELU) {
            p1 = 1; p2 = node->params[0]; p3 = node->params[0];
        } else if (node->op_type == BMK_OP_POS_ENC_ROPE) {
            p1 = 1; p2 = node->params[0]; p3 = node->params[0];
        }
        
        if (node->op_type == BMK_OP_ADD) {
            backend_encode_commit(enc);
            backend_encode_wait(enc);
            float* x = backend_buffer_map(sched->buffers[node->inputs[0]]);
            float* y = backend_buffer_map(sched->buffers[node->inputs[1]]);
            float* z = backend_buffer_map(sched->buffers[node->output]);
            int D = node->params[0];
            for(int d=0; d<D; d++) z[d] = x[d] + y[d];
            backend_buffer_unmap(sched->buffers[node->inputs[0]]);
            backend_buffer_unmap(sched->buffers[node->inputs[1]]);
            backend_buffer_unmap(sched->buffers[node->output]);
            enc = backend_encode_begin(sched->backend);
            continue;
        }

        backend_kernel_t* kn;
        if (node->op_type == BMK_OP_MATMUL && node->n_inputs >= 2
            && sched->graph->tensors[node->inputs[1]].quantized && sched->q8_kernel) {
            kn = sched->q8_kernel;   /* quantized weight -> Q8 dequantizing matmul */
        } else {
            kn = bmk_select(sched->reg, node->op_type, p1, p2, p3);
        }
        if (!kn) continue;
        
        int* p = backend_buffer_map(sched->param_bufs[i]);
        if (node->op_type == BMK_OP_MATMUL) {
            p[0] = node->params[0]; // BT
            p[1] = node->params[1]; // C  (in)
            p[2] = node->params[2]; // OC (out)
            p[3] = 0;               // has_bias (dummy zero bias)
        } else if (node->op_type == BMK_OP_FUSED_CLASSIFIER) {
            p[0] = node->params[1]; // C = D (in)   kernel reads C=p[0], V=p[1]
            p[1] = node->params[2]; // V     (out)
        } else {
            for(int j=0; j<node->n_params; j++) p[j] = node->params[j];
        }
        
        if (node->op_type == BMK_OP_ATTENTION) {
            p[4] = pos + 1; // seq_len to attend to
        } else if (node->op_type == BMK_OP_POS_ENC_ROPE) {
            int* p_pos = backend_buffer_map(sched->pos_bufs[i]);
            p_pos[0] = pos;
            backend_buffer_unmap(sched->pos_bufs[i]);
            
            float* ef = backend_buffer_map(sched->eps_bufs[i]);
            ef[0] = node->fparams[0];
            backend_buffer_unmap(sched->eps_bufs[i]);
            
            int* p_nkv = backend_buffer_map(sched->nkv_bufs[i]);
            p_nkv[0] = node->params[1]; // NKV heads
            backend_buffer_unmap(sched->nkv_bufs[i]);
            
            p[0] = node->params[0]; // HD
            p[1] = node->params[1]; // NKV
            p[2] = node->params[2]; // NH
        }
        
        backend_buffer_unmap(sched->param_bufs[i]);
        
        if (node->n_fparams > 0 && node->op_type != BMK_OP_POS_ENC_ROPE) {
            float* ef = backend_buffer_map(sched->eps_bufs[i]);
            for(int j=0; j<node->n_fparams; j++) ef[j] = node->fparams[j];
            backend_buffer_unmap(sched->eps_bufs[i]);
        }
        
        backend_buffer_t* bufs[12];
        int n_bufs = 0;
        backend_buffer_t* bout = sched->buffers[node->output];
        
        switch(node->op_type) {
            case BMK_OP_MATMUL: {
                bufs[0] = sched->buffers[node->inputs[0]]; // input
                bufs[1] = sched->buffers[node->inputs[1]]; // weight
                bufs[2] = sched->dummy_bias; // Dummy zero bias
                bufs[3] = bout; // output
                bufs[4] = sched->param_bufs[i]; // params
                n_bufs = 5;
                break;
            }
            case BMK_OP_NORM_RMS: {
                bufs[0] = sched->buffers[node->inputs[0]];
                bufs[1] = sched->buffers[node->inputs[1]];
                bufs[2] = bout;
                bufs[3] = sched->param_bufs[i];
                bufs[4] = sched->eps_bufs[i];
                n_bufs = 5;
                break;
            }
            case BMK_OP_NORM_LAYER: {
                bufs[0] = sched->buffers[node->inputs[0]];
                bufs[1] = sched->buffers[node->inputs[1]]; // weight
                bufs[2] = sched->buffers[node->inputs[2]]; // bias
                bufs[3] = bout;
                bufs[4] = sched->param_bufs[i];
                bufs[5] = sched->eps_bufs[i];
                n_bufs = 6;
                break;
            }
            case BMK_OP_ACT_SWIGLU: {
                bufs[0] = sched->buffers[node->inputs[0]];
                bufs[1] = sched->buffers[node->inputs[1]];
                bufs[2] = bout;
                bufs[3] = sched->param_bufs[i];
                n_bufs = 4;
                break;
            }
            case BMK_OP_ACT_GELU: {
                bufs[0] = sched->buffers[node->inputs[0]];
                bufs[1] = bout;
                bufs[2] = sched->param_bufs[i];
                n_bufs = 3;
                break;
            }
            case BMK_OP_POS_ENC_ROPE: {
                bufs[0] = sched->buffers[node->inputs[0]]; // q
                bufs[1] = sched->buffers[node->inputs[1]]; // k
                bufs[2] = sched->param_bufs[i]; // HD
                bufs[3] = sched->pos_bufs[i]; // pos
                bufs[4] = sched->eps_bufs[i]; // theta
                bufs[5] = sched->nkv_bufs[i]; // nkv
                n_bufs = 6;
                break;
            }
            case BMK_OP_ATTENTION: {
                // Update KV cache first!
                // node->inputs[1] is t_k, node->inputs[2] is t_v
                backend_buffer_t* bk = sched->buffers[node->inputs[1]];
                backend_buffer_t* bv = sched->buffers[node->inputs[2]];
                
                backend_encode_commit(enc); backend_encode_wait(enc); // Sync before CPU copy
                
                int KV = node->params[3];
                int l = node->params[6];
                int MS = sched->max_seq_len;
                float* kvc_gpu = (float*)backend_buffer_map(sched->bkvc);
                float* k_ptr = (float*)backend_buffer_map(bk);
                float* v_ptr = (float*)backend_buffer_map(bv);
                
                memcpy(kvc_gpu + l*2*MS*KV + pos*KV, k_ptr, KV*sizeof(float));
                memcpy(kvc_gpu + l*2*MS*KV + MS*KV + pos*KV, v_ptr, KV*sizeof(float));
                
                backend_buffer_unmap(bk); backend_buffer_unmap(bv); backend_buffer_unmap(sched->bkvc);
                
                enc = backend_encode_begin(sched->backend); // Resume
                
                bufs[0] = sched->buffers[node->inputs[0]]; // q
                bufs[1] = sched->bkvc;
                bufs[2] = bout;
                bufs[3] = sched->param_bufs[i];
                
                float* ef = backend_buffer_map(sched->eps_bufs[i]);
                memset(ef, 0, 16 * sizeof(float));
                ef[0] = 1.0f / sqrtf((float)node->params[1]); // scale
                backend_buffer_unmap(sched->eps_bufs[i]);
                
                bufs[4] = sched->eps_bufs[i];
                n_bufs = 5;
                break;
            }
            case BMK_OP_FUSED_RESIDUAL_NORM: {
                bufs[0] = sched->buffers[node->inputs[0]]; // res1
                bufs[1] = sched->buffers[node->inputs[1]]; // res2
                bufs[2] = sched->buffers[node->inputs[2]]; // wgt
                bufs[3] = bout;
                bufs[4] = sched->param_bufs[i];
                bufs[5] = sched->eps_bufs[i];
                n_bufs = 6;
                break;
            }
            case BMK_OP_FUSED_CLASSIFIER: {
                bufs[0] = sched->buffers[node->inputs[0]]; // x
                bufs[1] = sched->buffers[node->inputs[1]]; // lnfw
                bufs[2] = sched->buffers[node->inputs[2]]; // wcls
                bufs[3] = bout; // out
                bufs[4] = sched->param_bufs[i];
                bufs[5] = sched->eps_bufs[i];
                n_bufs = 6;
                break;
            }
            default:
                break;
        }
        
        int gtx = 1, gty = 1, gtz = 1;
        int ttx = 32, tty = 1, ttz = 1;
        
        if (node->op_type == BMK_OP_MATMUL) {
            gtx = (p3 + 31) & ~31;
            ttx = 32;
        } else if (node->op_type == BMK_OP_NORM_RMS || node->op_type == BMK_OP_FUSED_RESIDUAL_NORM || node->op_type == BMK_OP_NORM_LAYER) {
            gtx = 256;
            ttx = 256;
        } else if (node->op_type == BMK_OP_FUSED_CLASSIFIER) {
            gtx = (p3 + 255) & ~255;
            ttx = 256;
        } else if (node->op_type == BMK_OP_ACT_SWIGLU || node->op_type == BMK_OP_ACT_GELU) {
            gtx = p2;
            ttx = p2 < 256 ? p2 : 256;
        } else if (node->op_type == BMK_OP_POS_ENC_ROPE) {
            gtx = node->params[2]; // NH — one thread per head (loops over dim-pairs internally); covers K heads 0..NKV-1
            ttx = node->params[2] < 256 ? node->params[2] : 256;
        } else if (node->op_type == BMK_OP_ATTENTION) {
            gtx = node->params[0] * node->params[1]; // NH * HD
            ttx = node->params[1]; // HD
        }
        
        backend_encode_dispatch(enc, kn, bufs, NULL, n_bufs, gtx, gty, gtz, ttx, tty, ttz);
    }
    
    backend_encode_commit(enc);
    backend_encode_wait(enc);
}

void bmt_scheduler_set_input(bmt_scheduler_t* sched, int tensor_id, const void* data, size_t size) {
    if (!sched || tensor_id < 0 || tensor_id >= sched->graph->n_tensors || !data) return;
    void* dst = backend_buffer_map(sched->buffers[tensor_id]);
    memcpy(dst, data, size);
    backend_buffer_unmap(sched->buffers[tensor_id]);
}

void bmt_scheduler_get_output(bmt_scheduler_t* sched, int tensor_id, void* data, size_t size) {
    if (!sched || tensor_id < 0 || tensor_id >= sched->graph->n_tensors || !data) return;
    void* src = backend_buffer_map(sched->buffers[tensor_id]);
    memcpy(data, src, size);
    backend_buffer_unmap(sched->buffers[tensor_id]);
}
