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

static backend_kernel_t *km,*km_t,*kr,*kl,*kg,*ks,*kp,*ka;
static backend_buffer_t *bi,*bw,*bo,*bo2,*bp,*beps,*bbs;
static backend_ctx_t* g_be;
#define B() do{enc=backend_encode_begin(g_be);}while(0)
#define D(kn,bufs,n,gx,gy,gz,tx,ty,tz) do{backend_encode_dispatch(enc,kn,bufs,NULL,n,gx,gy,gz,tx,ty,tz);}while(0)
#define C() do{backend_encode_commit(enc);backend_encode_wait(enc);}while(0)

int bm_run_tokens(bm_context_t* ctx, bm_model_t* m,
                  const int* prompt_ids, int n_prompt,
                  int steps, float temperature, int top_k, float top_p,
                  uint64_t seed,
                  bm_token_cb_t callback, void* user_data);

static void softmax_inplace(float* x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

typedef struct { float val; int idx; } float_idx_t;

static int cmp_float_desc(const void* a, const void* b) {
    float fa = ((const float_idx_t*)a)->val;
    float fb = ((const float_idx_t*)b)->val;
    if (fa > fb) return -1;
    if (fa < fb) return 1;
    return 0;
}

static int sample_topk_topp(const float* logits, int n,
                            float temperature, int top_k, float top_p,
                            unsigned int rng_state) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float* probs = (float*)malloc(n * sizeof(float));
    if (!probs) return 0;
    for (int i = 0; i < n; i++) probs[i] = logits[i] / temperature;
    softmax_inplace(probs, n);

    if (top_k > 0 && top_k < n) {
        float_idx_t* sorted = (float_idx_t*)malloc(n * sizeof(float_idx_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(float_idx_t), cmp_float_desc);
            float thresh = sorted[top_k - 1].val;
            for (int i = 0; i < n; i++) if (probs[i] < thresh) probs[i] = 0.0f;
            free(sorted);
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += probs[i];
            if (s > 0.0f) for (int i = 0; i < n; i++) probs[i] /= s;
        }
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        float_idx_t* sorted = (float_idx_t*)malloc(n * sizeof(float_idx_t));
        if (sorted) {
            for (int i = 0; i < n; i++) { sorted[i].val = probs[i]; sorted[i].idx = i; }
            qsort(sorted, n, sizeof(float_idx_t), cmp_float_desc);
            float cum = 0.0f; float cut = 0.0f;
            for (int i = 0; i < n; i++) { cum += sorted[i].val; if (cum >= top_p) { cut = sorted[i].val; break; } }
            for (int i = 0; i < n; i++) if (probs[i] < cut) probs[i] = 0.0f;
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += probs[i];
            if (s > 0.0f) for (int i = 0; i < n; i++) probs[i] /= s;
            free(sorted);
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probs[i];
    unsigned int r = (rng_state * 1664525u + 1013904223u);
    r ^= r >> 13; r *= 1274126177u; r ^= r >> 16;
    float target = ((float)(r & 0xFFFFFF) / (float)0x1000000) * sum;
    float c = 0.0f;
    int pick = 0;
    for (int i = 0; i < n; i++) { c += probs[i]; if (c >= target) { pick = i; break; } }
    free(probs);
    return pick;
}

int bm_run(bm_context_t* ctx, bm_model_t* m, const char* prompt,
           int steps, float temp, unsigned long long seed, const char* tok_path) {
    g_be=ctx->backend_ctx;
    fprintf(stderr,"Loading tokenizer...\n"); fflush(stderr);
    bm_tokenizer_t tok; bm_tokenizer_init(&tok, (char*)tok_path, m->arch.vocab_size);
    fprintf(stderr,"Tokenizer loaded, encoding...\n"); fflush(stderr);
    int* ptok = (int*)malloc(1024 * sizeof(int));
    int nt = 0;
    bm_tokenizer_encode(&tok, prompt, 1, 0, ptok, &nt);
    int rc = bm_run_tokens(ctx, m, ptok, nt, steps, temp, 0, 1.0f, seed, NULL, NULL);
    free(ptok);
    bm_tokenizer_free(&tok);
    return rc;
}

int bm_run_tokens(bm_context_t* ctx, bm_model_t* m,
                  const int* prompt_ids, int n_prompt,
                  int steps, float temperature, int top_k, float top_p,
                  uint64_t seed,
                  bm_token_cb_t callback, void* user_data) {
    if (n_prompt <= 0 || !prompt_ids) return -1;
    g_be=ctx->backend_ctx;
    km=backend_kernel_create(g_be,"matmul_forward_naive");
    km_t=backend_kernel_create(g_be,"matmul_forward_tiled");
    kr=backend_kernel_create(g_be,"rmsnorm_forward");
    kl=backend_kernel_create(g_be,"layernorm_forward");
    kg=backend_kernel_create(g_be,"gelu_forward");
    ks=backend_kernel_create(g_be,"swiglu_forward");
    kp=backend_kernel_create(g_be,"rope_forward");
    ka=backend_kernel_create(g_be,"attention_forward");
    int D=m->arch.dim,H=m->arch.hidden_dim,NH=m->arch.n_heads,HD=m->head_size;
    int KV=m->kv_dim,KM=m->kv_mul,NKV=m->n_kv_heads,L=m->arch.n_layers,V=m->arch.vocab_size;
    int MS=m->arch.max_seq_len,nm=m->arch.norm,at=m->arch.activation,pt=m->arch.pos_enc;
    if (MS > 1024) MS = 1024;
    float sc=1.0f/sqrtf((float)HD);
    int max_dim=MAX(D,MAX(H,V));
    bi=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    bw=backend_buffer_alloc(g_be,MAX(V,3*NH*HD)*MAX(D,H)*sizeof(float));
    bo=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    bp=backend_buffer_alloc(g_be,8*sizeof(int));
    beps=backend_buffer_alloc(g_be,sizeof(float));
    bo2=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    bbs=backend_buffer_alloc(g_be,D*sizeof(float));
    *(float*)backend_buffer_map(beps)=1e-5f;

#define E2(outbuf,inp,wgt,woff,BT,CC,OC) do{ \
    int _p[]={BT,CC,OC,0}; memcpy(backend_buffer_map(bp),_p,4*sizeof(int)); \
    memcpy(backend_buffer_map(bi),(inp),BT*CC*sizeof(float)); \
    memcpy(backend_buffer_map(bw),(float*)(wgt)+(woff),OC*CC*sizeof(float)); \
    backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bp); \
    int _gtx=((OC)+31)&~31; \
    backend_buffer_t*_a[]={bi,bw,bbs,(outbuf),bp}; D(km_t,_a,5,_gtx,BT,1,32,1,1); \
}while(0)
#define E(inp,wgt,woff,BT,CC,OC) E2(bo,inp,wgt,woff,BT,CC,OC)
    #define R(inp,wgt) do{ \
        memcpy(backend_buffer_map(bi),(inp),D*sizeof(float)); \
        memcpy(backend_buffer_map(bw),(wgt),D*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){1,D},2*sizeof(int)); \
        backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bp); \
        backend_buffer_t*_n[]={bi,bw,bo,bp,beps}; D(kr,_n,5,1,1,1,1,1,1); \
    }while(0)
    #define L(inp,wgt,bias) do{ \
        memcpy(backend_buffer_map(bi),(inp),D*sizeof(float)); \
        memcpy(backend_buffer_map(bw),(wgt),D*sizeof(float)); \
        memcpy(backend_buffer_map(bbs),(bias),D*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){1,D},2*sizeof(int)); \
        backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bbs); backend_buffer_unmap(bp); \
        backend_buffer_t*_n[]={bi,bw,bo,bp,beps,bbs}; D(kl,_n,6,1,1,1,1,1,1); \
    }while(0)
    #define N(inp,wgt,bias) do{ if(nm==BM_NORM_LAYERNORM)L(inp,wgt,bias);else R(inp,wgt); }while(0)
    #define G(inp,N) do{ \
        memcpy(backend_buffer_map(bi),(inp),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){N},sizeof(int)); \
        backend_buffer_unmap(bi); backend_buffer_unmap(bp); \
        int _tx = (N) < 256 ? (N) : 256; \
        backend_buffer_t*_g[]={bi,bo,bp}; D(kg,_g,3,(N),1,1,_tx,1,1); \
    }while(0)
    #define S(gate,up,N) do{ \
        memcpy(backend_buffer_map(bi),(gate),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bw),(up),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){N},sizeof(int)); \
        backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bp); \
        int _tx = (N) < 256 ? (N) : 256; \
        backend_buffer_t*_s[]={bi,bw,bo,bp}; D(ks,_s,4,(N),1,1,_tx,1,1); \
    }while(0)

    float*x=calloc(D,sizeof(float)),*b=calloc(D,sizeof(float)),*logits=calloc(V,sizeof(float));
    float*hb=calloc(H,sizeof(float)),*hb2=calloc(H,sizeof(float));
    float*kvc=calloc(L*2*MS*KV,sizeof(float));
    backend_buffer_t* bkvc=backend_buffer_alloc(g_be,L*2*MS*KV*sizeof(float));
    backend_encoder_t*enc;
    backend_buffer_t *bq=backend_buffer_alloc(g_be,NH*HD*sizeof(float));
    backend_buffer_t *bk=backend_buffer_alloc(g_be,KV*sizeof(float));
    backend_buffer_t *bv=backend_buffer_alloc(g_be,KV*sizeof(float));
    backend_buffer_t *br=backend_buffer_alloc(g_be,4*sizeof(int));
    backend_buffer_t *bf=backend_buffer_alloc(g_be,sizeof(float));
    backend_buffer_t *btheta=backend_buffer_alloc(g_be,sizeof(float));
    backend_buffer_t *bnkv=backend_buffer_alloc(g_be,sizeof(int));
    *(float*)backend_buffer_map(btheta)=m->arch.rope_theta;
    *(int*)backend_buffer_map(bnkv)=m->n_kv_heads;
    if (seed == 0) seed = (uint64_t)time(NULL);
    unsigned int rng_state = (unsigned int)seed;

    int next_token = -1;
    for(int pos=0;pos<steps;pos++){
        int token;
        if (pos < n_prompt) {
            token = prompt_ids[pos];
        } else {
            token = next_token;
        }

        float*wte=m->token_embedding_table;
        memcpy(x,wte+token*D,D*sizeof(float));
        if (pos == 0) {
            FILE* ef = fopen("debug/embed.bin", "wb");
            if (ef) { fwrite(x, sizeof(float), D, ef); fclose(ef); }
        }
        if(m->arch.embed_scale){float esc=sqrtf((float)D);for(int i=0;i<D;i++)x[i]*=esc;}
        if(pt==BM_POS_LEARNED&&m->wpe)for(int i=0;i<D;i++)x[i]+=m->wpe[pos*D+i];

        for(int l=0;l<L;l++){
            B(); N(x,m->ln1w+l*D,m->ln1b+l*D); C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));

            float _q[NH*HD],_k[KV],_v[KV];
            B();E2(bq,b,m->qw+l*NH*HD*D,0,1,D,NH*HD);C();
            B();E2(bk,b,m->kw+l*NKV*HD*D,0,1,D,KV);C();
            B();E2(bv,b,m->vw+l*NKV*HD*D,0,1,D,KV);C();
            memcpy(_q,backend_buffer_map(bq),NH*HD*sizeof(float));
            memcpy(_k,backend_buffer_map(bk),KV*sizeof(float));
            memcpy(_v,backend_buffer_map(bv),KV*sizeof(float));

            if (pos == 0 && l == 0) {
                FILE* f = fopen("debug/l0_q.bin", "wb");
                if (f) { fwrite(_q, sizeof(float), NH*HD, f); fclose(f); }
                f = fopen("debug/l0_k.bin", "wb");
                if (f) { fwrite(_k, sizeof(float), KV, f); fclose(f); }
                f = fopen("debug/l0_v.bin", "wb");
                if (f) { fwrite(_v, sizeof(float), KV, f); fclose(f); }
            }

            if(m->arch.has_qk_norm){
                B();
                float* qnw=m->q_norm_w+l*HD;
                memcpy(backend_buffer_map(bi),_q,NH*HD*sizeof(float));
                for(int h=0;h<NH;h++)memcpy(backend_buffer_map(bw)+h*HD,qnw,HD*sizeof(float));
                memcpy(backend_buffer_map(bp),(int[]){NH,HD},2*sizeof(int));
                backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bp);
                backend_buffer_t*_qn[]={bi,bw,bo,bp,beps}; D(kr,_qn,5,NH,1,1,1,1,1);
                C();
                memcpy(_q,backend_buffer_map(bo),NH*HD*sizeof(float));
                B();
                float*knw=m->k_norm_w+l*m->n_kv_heads*HD;
                memcpy(backend_buffer_map(bi),_k,KV*sizeof(float));
                memcpy(backend_buffer_map(bw),knw,KV*sizeof(float));
                memcpy(backend_buffer_map(bp),(int[]){m->n_kv_heads,HD},2*sizeof(int));
                backend_buffer_unmap(bi); backend_buffer_unmap(bw); backend_buffer_unmap(bp);
                backend_buffer_t*_kn[]={bi,bw,bo,bp,beps}; D(kr,_kn,5,m->n_kv_heads,1,1,1,1,1);
                C();
                memcpy(_k,backend_buffer_map(bo),KV*sizeof(float));
                memcpy(backend_buffer_map(bq),_q,NH*HD*sizeof(float));
                memcpy(backend_buffer_map(bk),_k,KV*sizeof(float));
                backend_buffer_unmap(bq); backend_buffer_unmap(bk);
            }

            if(pt==BM_POS_ROPE){
                *(int*)backend_buffer_map(br)=HD;
                int ps=pos;
                *(int*)backend_buffer_map(bf)=ps;
                backend_buffer_unmap(br); backend_buffer_unmap(bf);
                B();
                backend_buffer_t *_rp[]={bq,bk,br,bf,btheta,bnkv};
                D(kp,_rp,6,MAX(NH,m->n_kv_heads),1,1,1,1,1);
                C();
                memcpy(_q,backend_buffer_map(bq),NH*HD*sizeof(float));
                memcpy(_k,backend_buffer_map(bk),KV*sizeof(float));
            }

            float*kk=kvc+l*2*MS*KV,*kv=kk+MS*KV;
            memcpy(kk+pos*KV,_k,KV*sizeof(float));
            memcpy(kv+pos*KV,_v,KV*sizeof(float));

            float* kvc_gpu = (float*)backend_buffer_map(bkvc);
            memcpy(kvc_gpu + l*2*MS*KV + pos*KV, _k, KV*sizeof(float));
            memcpy(kvc_gpu + l*2*MS*KV + MS*KV + pos*KV, _v, KV*sizeof(float));
            backend_buffer_unmap(bkvc);

            int S=pos+1;
            memcpy(backend_buffer_map(bq), _q, NH*HD*sizeof(float));
            backend_buffer_unmap(bq);

            int att_params[8] = {NH, HD, NKV, KV, S, KM, l, MS};
            memcpy(backend_buffer_map(bp), att_params, 8*sizeof(int));
            backend_buffer_unmap(bp);
            *(float*)backend_buffer_map(bf) = sc;
            backend_buffer_unmap(bf);

B();
            backend_buffer_t* att_bufs[] = {bq, bkvc, bo2, bp, bf};
            D(ka, att_bufs, 5, NH*256, 1, 1, 256, 1, 1);
            C();

            B();
            int _pp[]={1,NH*HD,D,0}; memcpy(backend_buffer_map(bp),_pp,4*sizeof(int));
            memcpy(backend_buffer_map(bw),m->attprojw+l*NH*HD*D,D*NH*HD*sizeof(float));
            backend_buffer_unmap(bw); backend_buffer_unmap(bp);
            backend_buffer_t* _pa[]={bo2,bw,bbs,bo,bp}; D(km,_pa,5,1,D,1,1,1,1);
            C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));

            for(int i=0;i<D;i++)x[i]+=b[i];
            if(m->arch.has_ffn_post_norm){
                B();N(x,m->ln2w+l*D,m->ln2b+l*D);C();
                memcpy(x,backend_buffer_map(bo),D*sizeof(float));
            }
            if(m->arch.has_ffn_post_norm){
                B();N(x,m->pre_ffn_w+l*D,NULL);C();
            }else{
                B();N(x,m->ln2w+l*D,m->ln2b+l*D);C();
            }
            
            if(m->arch.gated_mlp){
                float ffn_in[D];
                memcpy(ffn_in,backend_buffer_map(bo),D*sizeof(float));
                B();E(ffn_in,m->fcw+l*H*D,0,1,D,H);C();
                memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
                B();E(ffn_in,m->fcw3+l*H*D,0,1,D,H);C();
                memcpy(hb2,backend_buffer_map(bo),H*sizeof(float));
                if(at==BM_ACT_SWIGLU){
                    B();S(hb,hb2,H);C();
                    memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
                }else{
                    B();G(hb,H);C();
                    memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
                    for(int i=0;i<H;i++) hb[i] *= hb2[i];
                }
            }else{
                B();E(backend_buffer_map(bo),m->fcw+l*H*D,0,1,D,H);C();
                memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
                B();G(hb,H);C();
                memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
            }
            B();E(hb,m->fcprojw+l*D*H,0,1,H,D);C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));
            if(m->arch.has_ffn_post_norm){
                B();N(b,m->ffn_post_w+l*D,NULL);C();
                memcpy(b,backend_buffer_map(bo),D*sizeof(float));
            }
            for(int i=0;i<D;i++)x[i]+=b[i];
        }
        B();N(x,m->lnfw,m->lnfb);C();
        B();E(backend_buffer_map(bo),m->wcls,0,1,D,V);C();
        memcpy(logits,backend_buffer_map(bo),V*sizeof(float));

        if (pos < n_prompt + 3) {
            float max_logit = -INFINITY, min_logit = INFINITY;
            for (int i = 0; i < V; i++) {
                if (logits[i] > max_logit) max_logit = logits[i];
                if (logits[i] < min_logit) min_logit = logits[i];
            }
            fprintf(stderr, " [logits: min=%.2f max=%.2f]", min_logit, max_logit);

            float_idx_t* sorted = malloc(V * sizeof(float_idx_t));
            for (int i = 0; i < V; i++) { sorted[i].val = logits[i]; sorted[i].idx = i; }
            qsort(sorted, V, sizeof(float_idx_t), cmp_float_desc);
            fprintf(stderr, " [top5:");
            for (int i = 0; i < 5; i++) fprintf(stderr, " %d(%.2f)", sorted[i].idx, sorted[i].val);
            fprintf(stderr, "]");
            free(sorted);

            char fname[64];
            snprintf(fname, sizeof(fname), "debug/logits_pos%d.bin", pos);
            FILE* lf = fopen(fname, "wb");
            if (lf) { fwrite(logits, sizeof(float), V, lf); fclose(lf); }
        }

        if (pos >= n_prompt - 1) {
            next_token = sample_topk_topp(logits, V, temperature, top_k, top_p, rng_state++);
            if (callback) callback(next_token, user_data);
            if (pos < n_prompt + 5) {
                fprintf(stderr, " [gen token %d at pos %d]", next_token, pos);
            }
        }
        fprintf(stderr,"\r[step %d/%d]", pos + 1, steps);
        fflush(stderr);
    }
    printf("\n");
    backend_buffer_free(bq);backend_buffer_free(bk);backend_buffer_free(bv);
    backend_buffer_free(br);backend_buffer_free(bf);backend_buffer_free(bnkv);
    free(x);free(b);free(logits);free(hb);free(hb2);free(kvc);
    backend_buffer_free(bkvc);
    backend_buffer_free(bi);backend_buffer_free(bw);backend_buffer_free(bo);backend_buffer_free(bo2);backend_buffer_free(bp);backend_buffer_free(beps);
    backend_kernel_destroy(km);backend_kernel_destroy(kr);backend_kernel_destroy(kl);
    backend_kernel_destroy(kg);backend_kernel_destroy(ks);backend_kernel_destroy(kp);backend_kernel_destroy(ka);
    return 0;
}
