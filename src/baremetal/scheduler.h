#ifndef BMT_SCHEDULER_H
#define BMT_SCHEDULER_H

#include "baremetal/graph.h"
#include "kernels/registry.h"
#include "backend/backend.h"

typedef struct bmt_scheduler_s bmt_scheduler_t;

bmt_scheduler_t* bmt_scheduler_create(backend_ctx_t* backend, bmk_registry_t* reg, bmt_graph_t* graph);
void bmt_scheduler_destroy(bmt_scheduler_t* sched);
void bmt_scheduler_run(bmt_scheduler_t* sched, int pos, int seq_len);

#endif
