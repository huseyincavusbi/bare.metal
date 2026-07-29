#ifndef BMT_SCHEDULER_H
#define BMT_SCHEDULER_H

#include "baremetal/graph.h"
#include "kernels/registry.h"
#include "backend/backend.h"

typedef struct bmt_scheduler_s bmt_scheduler_t;

bmt_scheduler_t* bmt_scheduler_create(backend_ctx_t* backend, bmk_registry_t* reg, bmt_graph_t* graph, int max_seq_len, int kv_dim, int n_layers);
void bmt_scheduler_set_precision(bmt_scheduler_t* sched, bm_precision_t precision);
void bmt_scheduler_destroy(bmt_scheduler_t* sched);
void bmt_scheduler_run(bmt_scheduler_t* sched, int pos, int seq_len);
void bmt_scheduler_forward_train(bmt_scheduler_t* sched, int S);

/* Backward walk: reverse topological order, dispatch each node's backward
 * kernel using the forward activations (in buffers[]) + gradient buffers.
 * Gradient of the final output(s) must be seeded via bmt_scheduler_set_grad
 * (e.g. from xent_backward) before calling. Grad buffers are zeroed at
 * create; backward kernels that accumulate do so via atomics. */
void bmt_scheduler_backward(bmt_scheduler_t* sched);

/* Loss seed: dispatch xent_backward to fill grad_buffers[logits_id] from the
 * forward logits + targets. Returns the cross-entropy loss (for logging).
 * The trainer calls this between forward_train and backward. */
float bmt_scheduler_xent_backward(bmt_scheduler_t* sched, int logits_id, const int* targets, int S, int V);

void bmt_scheduler_set_input(bmt_scheduler_t* sched, int tensor_id, const void* data, size_t size);
void bmt_scheduler_get_output(bmt_scheduler_t* sched, int tensor_id, void* data, size_t size);
void bmt_scheduler_set_grad(bmt_scheduler_t* sched, int tensor_id, const void* data, size_t size);
void bmt_scheduler_get_grad(bmt_scheduler_t* sched, int tensor_id, void* data, size_t size);

/* Direct buffer access for the trainer (embedding_backward / adamw dispatch
 * need raw backend_buffer_t* to the forward activation and grad buffers).
 * Keeps the scheduler struct opaque to everyone else. */
backend_buffer_t* bmt_scheduler_get_buffer(bmt_scheduler_t* sched, int tensor_id);
backend_buffer_t* bmt_scheduler_get_grad_buffer(bmt_scheduler_t* sched, int tensor_id);

#endif
