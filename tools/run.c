#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/context.h"
#include "baremetal/tokenizer.h"
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/param.h>

#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

static backend_kernel_t* kn_matmul;
static backend_kernel_t* kn_rmsnorm;

static void bmt_run_init_kernels(backend_ctx_t* be) {
    kn_matmul  = backend_kernel_create(be, "matmul_forward_naive");
    kn_rmsnorm = backend_kernel_create(be, "rmsnorm_forward");
}

static void bmt_run_free_kernels(void) {
    backend_kernel_destroy(kn_matmul);
    backend_kernel_destroy(kn_rmsnorm);
}

int bm_run(bm_context_t* ctx, bm_model_t* model, const char* prompt,
           int steps, float temperature, unsigned long long seed) {
    (void)seed;
    backend_ctx_t* be = ctx->backend_ctx;
    bmt_run_init_kernels(be);
    int D = model->arch.dim, H = model->arch.hidden_dim;
    int NH = model->arch.n_heads, HD = model->head_size;
    int KV_DIM = model->kv_dim, KV_MUL = model->kv_mul;
    int L = model->arch.n_layers;
    int V = model->arch.vocab_size;
    int max_seq = model->arch.max_seq_len;
    float scale = 1.0f / sqrtf((float)HD);
    int norm_t = model->arch.norm;
    int act_t = model->arch.activation;
    int rope = model->arch.pos_enc == BM_POS_ROPE;
    (void)act_t;

    float* w = (float*)model->weight_buffer;
    (void)w;
    float* x  = calloc(D, sizeof(float));
    float* xb = calloc(D, sizeof(float));
    float* logits = calloc(V, sizeof(float));

    int max_dim = MAX(D, MAX(H, V));
    backend_buffer_t* b_in  = backend_buffer_alloc(be, max_dim * sizeof(float));
    backend_buffer_t* b_w   = backend_buffer_alloc(be, MAX(V, 3*NH*HD) * MAX(D, H) * sizeof(float));
    backend_buffer_t* b_out = backend_buffer_alloc(be, MAX(D, MAX(H, MAX(V, NH*HD))) * sizeof(float));
    backend_buffer_t* b_p4  = backend_buffer_alloc(be, 4*sizeof(int));
    backend_buffer_t* b_eps  = backend_buffer_alloc(be, sizeof(float));
    *(float*)backend_buffer_map(b_eps) = 1e-5f;

    #define MTL_MATMUL(inp, wgt, woff, BT, CC, OC, out) do { \
        memcpy(backend_buffer_map(b_in), (inp), BT*CC*sizeof(float)); \
        memcpy(backend_buffer_map(b_w), (float*)(wgt)+(woff), OC*CC*sizeof(float)); \
        memcpy(backend_buffer_map(b_p4), (int[]){BT,CC,OC,0}, 4*sizeof(int)); \
        backend_buffer_t* _a[] = {b_in, b_w, b_in, b_out, b_p4}; \
        backend_kernel_dispatch(be, kn_matmul, _a, NULL, 5, BT, OC, 1, 1, 1, 1); \
        memcpy((out), backend_buffer_map(b_out), BT*OC*sizeof(float)); \
    } while(0)

    #define MTL_RMSNORM(inp, wgt, out) do { \
        memcpy(backend_buffer_map(b_in), (inp), D*sizeof(float)); \
        memcpy(backend_buffer_map(b_w), (wgt), D*sizeof(float)); \
        memcpy(backend_buffer_map(b_p4), (int[]){1,D}, 2*sizeof(int)); \
        backend_buffer_t* _n[] = {b_in, b_w, b_out, b_p4, b_eps}; \
        backend_kernel_dispatch(be, kn_rmsnorm, _n, NULL, 5, 1, 1, 1, 1, 1, 1); \
        memcpy((out), backend_buffer_map(b_out), D*sizeof(float)); \
    } while(0)

    // KV cache: [L][2][max_seq][KV_DIM]
    float* kv_cache = calloc(L * 2 * max_seq * KV_DIM, sizeof(float));

    // BPE tokenizer
    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, "tokenizer.bin", model->arch.vocab_size);
    int* prompt_tokens = malloc(256 * sizeof(int));
    int num_tokens = 0;
    bm_tokenizer_encode(&tok, prompt, 1, 0, prompt_tokens, &num_tokens);

    int token = prompt_tokens[0];
    int next;

    for (int pos = 0; pos < steps; pos++) {
        // embed
        float* wte = model->token_embedding_table;
        memcpy(x, wte + token * D, D * sizeof(float));
        if (model->wpe)
            for (int i = 0; i < D; i++) x[i] += model->wpe[pos * D + i];

        for (int l = 0; l < L; l++) {
            // attn norm
            if (norm_t == BM_NORM_RMSNORM) MTL_RMSNORM(x, model->ln1w + l*D, xb);
            else memcpy(xb, x, D*sizeof(float));

            // QKV projection (CPU for simplicity)
            float* qkv_w = model->qkvw + l * 3 * D * NH * HD;
            float* q = calloc(NH*HD, sizeof(float));
            float* k = calloc(KV_DIM, sizeof(float));
            float* v = calloc(KV_DIM, sizeof(float));
            for (int i = 0; i < NH*HD; i++) {
                float s = 0;
                for (int j = 0; j < D; j++) s += xb[j] * qkv_w[i*D+j];
                q[i] = s;
            }
            for (int i = 0; i < KV_DIM; i++) {
                k[i] = q[NH*HD + i];
                v[i] = q[NH*HD + KV_DIM + i];
            }

            // RoPE
            if (rope) {
                for (int i = 0; i < HD; i += 2) {
                    float freq = 1.0f/powf(10000.0f, (float)i/(float)HD);
                    float c = cosf((float)pos*freq), s = sinf((float)pos*freq);
                    for (int h = 0; h < NH; h++) {
                        float q0=q[h*HD+i], q1=q[h*HD+i+1];
                        q[h*HD+i]=q0*c-q1*s; q[h*HD+i+1]=q0*s+q1*c;
                    }
                    for (int hh = 0; hh < model->n_kv_heads; hh++) {
                        float k0=k[hh*HD+i], k1=k[hh*HD+i+1];
                        k[hh*HD+i]=k0*c-k1*s; k[hh*HD+i+1]=k0*s+k1*c;
                    }
                }
            }

            // Store in KV cache
            float* kc_k = kv_cache + l * 2 * max_seq * KV_DIM;
            float* kc_v = kc_k + max_seq * KV_DIM;
            memcpy(kc_k + pos*KV_DIM, k, KV_DIM*sizeof(float));
            memcpy(kc_v + pos*KV_DIM, v, KV_DIM*sizeof(float));

            // Attention per head
            float* att = calloc(NH*(pos+1), sizeof(float));
            int S = pos + 1;
            float* xb_att = calloc(NH*HD, sizeof(float));
            for (int h = 0; h < NH; h++) {
                int kv_h = h / KV_MUL;
                for (int t = 0; t < S; t++) {
                    float d = 0;
                    for (int i = 0; i < HD; i++)
                        d += q[h*HD+i] * kc_k[kv_h*KV_DIM + t*KV_DIM + i];
                    att[h*S+t] = d * scale;
                }
                float m = -INFINITY; float sum = 0;
                for (int t = 0; t < S; t++) {
                    if (att[h*S+t] > m) m = att[h*S+t];
                }
                for (int t = 0; t < S; t++) {
                    att[h*S+t] = expf(att[h*S+t]-m); sum += att[h*S+t];
                }
                for (int t = 0; t < S; t++) att[h*S+t] /= sum;

                for (int i = 0; i < HD; i++) {
                    float d = 0;
                    for (int t = 0; t < S; t++)
                        d += att[h*S+t] * kc_v[kv_h*KV_DIM + t*KV_DIM + i];
                    xb_att[h*HD+i] = d;
                }
            }
            free(att);

            // Output projection
            MTL_MATMUL(xb_att, model->attprojw + l*NH*HD*D, 0, 1, NH*HD, D, xb);
            free(xb_att);

            // Residual
            for (int i = 0; i < D; i++) x[i] += xb[i];

            // FFN norm
            if (norm_t == BM_NORM_RMSNORM) MTL_RMSNORM(x, model->ln2w + l*D, xb);
            else memcpy(xb, x, D*sizeof(float));

            // FFN gate + up + activation
            float* hb = calloc(H, sizeof(float));
            float* hb2 = calloc(H, sizeof(float));
            MTL_MATMUL(xb, model->fcw + l*H*D, 0, 1, D, H, hb);
            MTL_MATMUL(xb, model->fcw3 + l*H*D, 0, 1, D, H, hb2);
            for (int i = 0; i < H; i++) {
                float sx = hb[i] / (1.0f + expf(-hb[i]));
                hb[i] = sx * hb2[i];
            }

            // FFN down
            MTL_MATMUL(hb, model->fcprojw + l*D*H, 0, 1, H, D, xb);
            free(hb); free(hb2);

            // Residual
            for (int i = 0; i < D; i++) x[i] += xb[i];
            free(q); free(k); free(v);
        }

        // Final norm + classifier
        if (norm_t == BM_NORM_RMSNORM) MTL_RMSNORM(x, model->lnfw, xb);
        else memcpy(xb, x, D*sizeof(float));
        MTL_MATMUL(xb, model->wcls, 0, 1, D, V, logits);

        // Sample
        if (pos < num_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            if (temperature == 0) {
                next = 0; float mv = logits[0];
                for (int i = 1; i < V; i++) if (logits[i] > mv) { mv = logits[i]; next = i; }
            } else {
                float sum = 0;
                for (int i = 0; i < V; i++) { logits[i] = expf(logits[i]/temperature); sum += logits[i]; }
                float r = (float)rand() / (float)RAND_MAX * sum;
                float c = 0;
                for (next = 0; next < V; next++) { c += logits[next]; if (c >= r) break; }
            }
        }

        if (next > 31 && next < 127) putchar(next);
        token = next;
    }
    printf("\n");

    free(prompt_tokens);
    bm_tokenizer_free(&tok);
    backend_buffer_free(b_in); backend_buffer_free(b_w); backend_buffer_free(b_out);
    backend_buffer_free(b_p4); backend_buffer_free(b_eps);
    bmt_run_free_kernels();
    return 0;
}
