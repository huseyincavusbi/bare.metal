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
static backend_kernel_t* kn_layernorm;
static backend_kernel_t* kn_gelu;
static backend_kernel_t* kn_swiglu;
static void init_k(backend_ctx_t* be) {
    kn_matmul   = backend_kernel_create(be, "matmul_forward_naive");
    kn_rmsnorm  = backend_kernel_create(be, "rmsnorm_forward");
    kn_layernorm= backend_kernel_create(be, "layernorm_forward");
    kn_gelu     = backend_kernel_create(be, "gelu_forward");
    kn_swiglu   = backend_kernel_create(be, "swiglu_forward");
}
static void free_k(void) {
    backend_kernel_destroy(kn_matmul); backend_kernel_destroy(kn_rmsnorm);
    backend_kernel_destroy(kn_layernorm); backend_kernel_destroy(kn_gelu);
    backend_kernel_destroy(kn_swiglu);
}

int bm_run(bm_context_t* ctx, bm_model_t* model, const char* prompt,
           int steps, float temperature, unsigned long long seed) {
    backend_ctx_t* be = ctx->backend_ctx;
    init_k(be);
    int D=model->arch.dim,H=model->arch.hidden_dim,NH=model->arch.n_heads,HD=model->head_size;
    int KV_DIM=model->kv_dim,KV_MUL=model->kv_mul,L=model->arch.n_layers,V=model->arch.vocab_size;
    int max_seq=model->arch.max_seq_len,norm_t=model->arch.norm,act_t=model->arch.activation;
    int pos_t=model->arch.pos_enc;
    float scale=1.0f/sqrtf((float)HD);
    (void)seed;

    float *x=calloc(D,sizeof(float)),*xb=calloc(D,sizeof(float)),*logits=calloc(V,sizeof(float));
    float *hb=calloc(H,sizeof(float)),*hb2=calloc(H,sizeof(float));
    int max_dim=MAX(D,MAX(H,V));
    backend_buffer_t *b_in=backend_buffer_alloc(be,max_dim*sizeof(float));
    backend_buffer_t *b_w=backend_buffer_alloc(be,MAX(V,3*NH*HD)*MAX(D,H)*sizeof(float));
    backend_buffer_t *b_out=backend_buffer_alloc(be,max_dim*sizeof(float));
    backend_buffer_t *b_p4=backend_buffer_alloc(be,4*sizeof(int));
    backend_buffer_t *b_eps=backend_buffer_alloc(be,sizeof(float));
    *(float*)backend_buffer_map(b_eps)=1e-5f;

#define M(inp,wgt,woff,BT,CC,OC,out) do { \
    memcpy(backend_buffer_map(b_in),(inp),BT*CC*sizeof(float)); \
    memcpy(backend_buffer_map(b_w),(float*)(wgt)+(woff),OC*CC*sizeof(float)); \
    memcpy(backend_buffer_map(b_p4),(int[]){BT,CC,OC,0},4*sizeof(int)); \
    backend_buffer_t*_a[]={b_in,b_w,b_in,b_out,b_p4}; \
    backend_kernel_dispatch(be,kn_matmul,_a,NULL,5,BT,OC,1,1,1,1); \
    memcpy((out),backend_buffer_map(b_out),BT*OC*sizeof(float)); \
}while(0)
#define N(inp,wgt,out) do { \
    backend_kernel_t*kn=(norm_t==BM_NORM_RMSNORM)?kn_rmsnorm:kn_layernorm; \
    memcpy(backend_buffer_map(b_in),(inp),D*sizeof(float)); \
    memcpy(backend_buffer_map(b_w),(wgt),D*sizeof(float)); \
    memcpy(backend_buffer_map(b_p4),(int[]){1,D},2*sizeof(int)); \
    backend_buffer_t*_n[]={b_in,b_w,b_out,b_p4,b_eps}; \
    backend_kernel_dispatch(be,kn,_n,NULL,5,1,1,1,1,1,1); \
    memcpy((out),backend_buffer_map(b_out),D*sizeof(float)); \
}while(0)
#define G(inp,N) do { \
    memcpy(backend_buffer_map(b_in),(inp),(N)*sizeof(float)); \
    memcpy(backend_buffer_map(b_p4),(int[]){N},sizeof(int)); \
    backend_buffer_t*_g[]={b_in,b_out,b_p4}; \
    backend_kernel_dispatch(be,kn_gelu,_g,NULL,3,(N),1,1,(N),1,1); \
    memcpy((inp),backend_buffer_map(b_out),(N)*sizeof(float)); \
}while(0)
#define S(gate,up,N) do { \
    memcpy(backend_buffer_map(b_in),(gate),(N)*sizeof(float)); \
    memcpy(backend_buffer_map(b_w),(up),(N)*sizeof(float)); \
    memcpy(backend_buffer_map(b_p4),(int[]){N},sizeof(int)); \
    backend_buffer_t*_s[]={b_in,b_w,b_out,b_p4}; \
    backend_kernel_dispatch(be,kn_swiglu,_s,NULL,4,(N),1,1,(N),1,1); \
    memcpy((gate),backend_buffer_map(b_out),(N)*sizeof(float)); \
}while(0)

    float* kv_cache = calloc(L*2*max_seq*KV_DIM, sizeof(float));
    bm_tokenizer_t tok;
    bm_tokenizer_init(&tok, "tokenizer.bin", V);
    int* prompt_tokens = malloc(1024*sizeof(int));
    int num_tokens=0;
    bm_tokenizer_encode(&tok, prompt, 1, 0, prompt_tokens, &num_tokens);

    int token=prompt_tokens[0], next, prev_token=1;
    for(int pos=0; pos<steps; pos++) {
        // Embed
        float* wte=model->token_embedding_table;
        memcpy(x, wte+token*D, D*sizeof(float));
        if(pos_t==BM_POS_LEARNED && model->wpe)
            for(int i=0;i<D;i++) x[i]+=model->wpe[pos*D+i];

        for(int l=0; l<L; l++) {
            N(x, model->ln1w+l*D, xb);

            // QKV
            float *q=calloc(NH*HD,sizeof(float)),*k=calloc(KV_DIM,sizeof(float)),*v=calloc(KV_DIM,sizeof(float));
            M(xb, model->qkvw+l*3*D*NH*HD, 0, 1, D, 3*(NH*HD), q);
            for(int i=0;i<KV_DIM;i++) { k[i]=q[NH*HD+i]; v[i]=q[NH*HD+KV_DIM+i]; }

            // RoPE
            if(pos_t==BM_POS_ROPE) {
                for(int i=0;i<HD;i+=2) {
                    float freq=1.0f/powf(10000.0f,(float)i/(float)HD);
                    float c=cosf((float)pos*freq),s=sinf((float)pos*freq);
                    for(int h=0;h<NH;h++) {
                        float q0=q[h*HD+i],q1=q[h*HD+i+1];
                        q[h*HD+i]=q0*c-q1*s; q[h*HD+i+1]=q0*s+q1*c;
                    }
                    for(int hh=0;hh<model->n_kv_heads;hh++) {
                        float k0=k[hh*HD+i],k1=k[hh*HD+i+1];
                        k[hh*HD+i]=k0*c-k1*s; k[hh*HD+i+1]=k0*s+k1*c;
                    }
                }
            }

            // KV cache
            float *kc_k=kv_cache+l*2*max_seq*KV_DIM, *kc_v=kc_k+max_seq*KV_DIM;
            memcpy(kc_k+pos*KV_DIM, k, KV_DIM*sizeof(float));
            memcpy(kc_v+pos*KV_DIM, v, KV_DIM*sizeof(float));

            // Attention: GPU matmul for QK^T and SV
            int S=pos+1;
            float* xb_att=calloc(NH*HD,sizeof(float));
            for(int h=0;h<NH;h++) {
                int kv_h=h/KV_MUL;
                float *qh=q+h*HD, *kbuf=calloc(S*HD,sizeof(float)), *vbuf=calloc(S*HD,sizeof(float));
                for(int t=0;t<S;t++) for(int i=0;i<HD;i++) {
                    kbuf[t*HD+i]=kc_k[kv_h*KV_DIM+t*KV_DIM+i];
                    vbuf[t*HD+i]=kc_v[kv_h*KV_DIM+t*KV_DIM+i];
                }
                float *scores=calloc(S,sizeof(float)), *h_out=calloc(HD,sizeof(float));
                M(qh, kbuf, 0, 1, HD, S, scores);
                for(int t=0;t<S;t++) scores[t]*=scale;
                float m=-INFINITY,sum=0;
                for(int t=0;t<S;t++) if(scores[t]>m) m=scores[t];
                for(int t=0;t<S;t++) { scores[t]=expf(scores[t]-m); sum+=scores[t]; }
                for(int t=0;t<S;t++) scores[t]/=sum;
                M(scores, vbuf, 0, 1, S, HD, h_out);
                for(int i=0;i<HD;i++) xb_att[h*HD+i]=h_out[i];
                free(kbuf); free(vbuf); free(scores); free(h_out);
            }

            // Output proj + residual
            M(xb_att, model->attprojw+l*NH*HD*D, 0, 1, NH*HD, D, xb);
            free(xb_att);
            for(int i=0;i<D;i++) x[i]+=xb[i];

            // FFN norm
            N(x, model->ln2w+l*D, xb);

            // FFN
            M(xb, model->fcw+l*H*D, 0, 1, D, H, hb);
            if(act_t==BM_ACT_SWIGLU) {
                M(xb, model->fcw3+l*H*D, 0, 1, D, H, hb2);
                S(hb, hb2, H);
            } else {
                G(hb, H);
            }
            M(hb, model->fcprojw+l*D*H, 0, 1, H, D, xb);
            for(int i=0;i<D;i++) x[i]+=xb[i];
            free(q); free(k); free(v);
        }

        // Final norm + classifier
        N(x, model->lnfw, xb);
        M(xb, model->wcls, 0, 1, D, V, logits);

        // Sample
        if(pos<num_tokens-1) next=prompt_tokens[pos+1];
        else {
            if(temperature==0) { next=0; float mv=logits[0]; for(int i=1;i<V;i++) if(logits[i]>mv){mv=logits[i];next=i;} }
            else { float sum=0; for(int i=0;i<V;i++){logits[i]=expf(logits[i]/temperature);sum+=logits[i];}
                   float r=(float)rand()/(float)RAND_MAX*sum,c=0;
                   for(next=0;next<V;next++){c+=logits[next];if(c>=r)break;} }
        }
        bm_tokenizer_safe_print(bm_tokenizer_decode(&tok, prev_token, next));
        prev_token=token; token=next;
    }
    printf("\n");

    free(prompt_tokens); bm_tokenizer_free(&tok); free(kv_cache);
    free(x); free(xb); free(logits); free(hb); free(hb2);
    backend_buffer_free(b_in); backend_buffer_free(b_w); backend_buffer_free(b_out);
    backend_buffer_free(b_p4); backend_buffer_free(b_eps);
    free_k(); return 0;
}
