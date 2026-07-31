// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/compiler.h"
#include "utils/log.h"
#include <stddef.h>

void bmt_compiler_run(bmt_graph_t* graph) {
    if (!graph) return;
    int fused_count = 0;

    // Pattern 1: ADD + NORM_RMS -> FUSED_RESIDUAL_NORM
    for (int i = 0; i < graph->n_nodes; i++) {
        bmt_node_t* norm_node = &graph->nodes[i];
        if (norm_node->op_type == BMK_OP_NORM_RMS) {
            int in_tensor_id = norm_node->inputs[0];

            // Find the ADD that produces this norm's input. The residual
            // stream tensor (t_x) is written in-place by every layer's ADD,
            // so we must pick the NEAREST preceding ADD (scanning backwards),
            // not the first node whose output matches. Skip dead nodes.
            bmt_node_t* add_node = NULL;
            for (int j = i - 1; j >= 0; j--) {
                bmt_node_t* cand = &graph->nodes[j];
                if (cand->op_type == BMK_OP_COUNT) continue;
                if (cand->output == in_tensor_id && cand->op_type == BMK_OP_ADD) {
                    add_node = cand;
                    break;
                }
            }

            if (add_node) {
                /* Only fuse if the ADD's output is used solely by this NORM.
                 * In training graph, t_res1 is also read by the second ADD. */
                int res_usage = 0;
                for (int j = 0; j < graph->n_nodes; j++) {
                    bmt_node_t* c = &graph->nodes[j];
                    if (c->op_type == BMK_OP_COUNT) continue;
                    for (int k = 0; k < c->n_inputs; k++)
                        if (c->inputs[k] == in_tensor_id) res_usage++;
                }
                if (res_usage != 1) continue;

                /* Don't fuse the final norm (before classifier) — it has no
                 * preceding ADD in the training graph; the ADD is the second
                 * residual of the last layer, and its inputs may be overwritten. */
                if (i == graph->n_nodes - 2) continue;

                int x_id = add_node->inputs[0];
                int add_val_id = add_node->inputs[1];
                int weight_id = norm_node->inputs[1];

                norm_node->op_type = BMK_OP_FUSED_RESIDUAL_NORM;
                norm_node->n_inputs = 3;
                norm_node->inputs[0] = x_id;
                norm_node->inputs[1] = add_val_id;
                norm_node->inputs[2] = weight_id;

                add_node->op_type = BMK_OP_COUNT; // Dead node
                fused_count++;
            }
        }
    }

    // Pattern 2: NORM_RMS + MATMUL -> FUSED_CLASSIFIER
    for (int i = 0; i < graph->n_nodes; i++) {
        bmt_node_t* mm_node = &graph->nodes[i];
        if (mm_node->op_type == BMK_OP_MATMUL) {
            int in_tensor_id = mm_node->inputs[0];
            
            bmt_node_t* norm_node = NULL;
            for (int j = i - 1; j >= 0; j--) {
                bmt_node_t* cand = &graph->nodes[j];
                if (cand->op_type == BMK_OP_COUNT) continue;
                if (cand->output == in_tensor_id && cand->op_type == BMK_OP_NORM_RMS) {
                    norm_node = cand;
                    break;
                }
            }

            if (norm_node) {
                int usage_count = 0;
                for (int j = 0; j < graph->n_nodes; j++) {
                    for (int k = 0; k < graph->nodes[j].n_inputs; k++) {
                        if (graph->nodes[j].inputs[k] == in_tensor_id) usage_count++;
                    }
                }
                
                if (usage_count == 1) {
                    int x_id = norm_node->inputs[0];
                    int lnfw_id = norm_node->inputs[1];
                    int wcls_id = mm_node->inputs[1];

                    mm_node->op_type = BMK_OP_FUSED_CLASSIFIER;
                    mm_node->n_inputs = 3;
                    mm_node->inputs[0] = x_id;
                    mm_node->inputs[1] = lnfw_id;
                    mm_node->inputs[2] = wcls_id;

                    mm_node->n_fparams = norm_node->n_fparams;
                    for (int k=0; k<norm_node->n_fparams; k++) mm_node->fparams[k] = norm_node->fparams[k];

                    norm_node->op_type = BMK_OP_COUNT; // Dead node
                    fused_count++;
                }
            }
        }
    }

    BMT_LOG_INFO("Graph compiler applied %d fusion passes", fused_count);
}
