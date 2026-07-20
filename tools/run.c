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

static backend_kernel_t *km,*kr,*kl,*kg,*ks,*kp;
static backend_buffer_t *bi,*bw,*bo,*bo2,*bp,*beps;
static backend_ctx_t* g_be;
#define B() do{enc=backend_encode_begin(g_be);}while(0)
#define D(kn,bufs,n,gx,gy,gz,tx,ty,tz) do{backend_encode_dispatch(enc,kn,bufs,NULL,n,gx,gy,gz,tx,ty,tz);}while(0)
#define C() do{backend_encode_commit(enc);backend_encode_wait(enc);}while(0)

int bm_run(bm_context_t* ctx, bm_model_t* m, const char* prompt,
           int steps, float temp, unsigned long long seed, const char* tok_path) {
    (void)seed;
    g_be=ctx->backend_ctx;
    km=backend_kernel_create(g_be,"matmul_forward_naive");
    kr=backend_kernel_create(g_be,"rmsnorm_forward");
    kl=backend_kernel_create(g_be,"layernorm_forward");
    kg=backend_kernel_create(g_be,"gelu_forward");
    ks=backend_kernel_create(g_be,"swiglu_forward");
    kp=backend_kernel_create(g_be,"rope_forward");
    backend_kernel_t* ksx = backend_kernel_create(g_be,"softmax_causal_scale");
    (void)ksx; // unused for now
    int D=m->arch.dim,H=m->arch.hidden_dim,NH=m->arch.n_heads,HD=m->head_size;
    int KV=m->kv_dim,KM=m->kv_mul,NKV=m->n_kv_heads,L=m->arch.n_layers,V=m->arch.vocab_size;
    int MS=m->arch.max_seq_len,nm=m->arch.norm,at=m->arch.activation,pt=m->arch.pos_enc;
    if (MS > 1024) MS = 1024; // cap to avoid OOM
    float sc=1.0f/sqrtf((float)HD); (void)seed;
    int max_dim=MAX(D,MAX(H,V));
    bi=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    bw=backend_buffer_alloc(g_be,MAX(V,3*NH*HD)*MAX(D,H)*sizeof(float));
    bo=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    bp=backend_buffer_alloc(g_be,4*sizeof(int));
    beps=backend_buffer_alloc(g_be,sizeof(float));
    bo2=backend_buffer_alloc(g_be,max_dim*sizeof(float));
    *(float*)backend_buffer_map(beps)=1e-5f;

#define E2(outbuf,inp,wgt,woff,BT,CC,OC) do{ \
    int _p[]={BT,CC,OC,0}; memcpy(backend_buffer_map(bp),_p,4*sizeof(int)); \
    memcpy(backend_buffer_map(bi),(inp),BT*CC*sizeof(float)); \
    memcpy(backend_buffer_map(bw),(float*)(wgt)+(woff),OC*CC*sizeof(float)); \
    backend_buffer_t*_a[]={bi,bw,bi,(outbuf),bp}; D(km,_a,5,BT,OC,1,1,1,1); \
}while(0)
#define E(inp,wgt,woff,BT,CC,OC) E2(bo,inp,wgt,woff,BT,CC,OC)
    #define R(inp,wgt) do{ \
        memcpy(backend_buffer_map(bi),(inp),D*sizeof(float)); \
        memcpy(backend_buffer_map(bw),(wgt),D*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){1,D},2*sizeof(int)); \
        backend_kernel_t*kn=(nm==BM_NORM_RMSNORM)?kr:kl; \
        backend_buffer_t*_n[]={bi,bw,bo,bp,beps}; D(kn,_n,5,1,1,1,1,1,1); \
    }while(0)
    #define G(inp,N) do{ \
        memcpy(backend_buffer_map(bi),(inp),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){N},sizeof(int)); \
        backend_buffer_t*_g[]={bi,bo,bp}; D(kg,_g,3,(N),1,1,(N),1,1); \
    }while(0)
    #define S(gate,up,N) do{ \
        memcpy(backend_buffer_map(bi),(gate),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bw),(up),(N)*sizeof(float)); \
        memcpy(backend_buffer_map(bp),(int[]){N},sizeof(int)); \
        backend_buffer_t*_s[]={bi,bw,bo,bp}; D(ks,_s,4,(N),1,1,(N),1,1); \
    }while(0)
    #define RP(qbuf,kbuf,hdim,pos) do{ \
        *(int*)backend_buffer_map(bp)=hdim; \
        int _p=pos; \
        backend_buffer_t*_rp[3]; _rp[0]=qbuf; _rp[1]=kbuf; \
        _rp[2]=backend_buffer_alloc(g_be,sizeof(float)); \
        memcpy(backend_buffer_map(_rp[2]),&_p,0); /* won't work */ \
    }while(0)

    float*x=calloc(D,sizeof(float)),*b=calloc(D,sizeof(float)),*logits=calloc(V,sizeof(float));
    float*hb=calloc(H,sizeof(float)),*hb2=calloc(H,sizeof(float));
    float*kvc=calloc(L*2*MS*KV,sizeof(float));
    fprintf(stderr,"Loading tokenizer...\n"); fflush(stderr);
    bm_tokenizer_t tok; bm_tokenizer_init(&tok, (char*)tok_path, V);
    fprintf(stderr,"Tokenizer loaded, encoding...\n"); fflush(stderr);
    int*ptok=malloc(1024*sizeof(int)); int nt=0; int prev=1;
    bm_tokenizer_encode(&tok,prompt,1,0,ptok,&nt);
    int token=ptok[0],next;
    backend_encoder_t*enc;
    backend_buffer_t *bq=backend_buffer_alloc(g_be,NH*HD*sizeof(float));
    backend_buffer_t *bk=backend_buffer_alloc(g_be,KV*sizeof(float));
    backend_buffer_t *bv=backend_buffer_alloc(g_be,KV*sizeof(float));
    backend_buffer_t *br=backend_buffer_alloc(g_be,4*sizeof(int));
    backend_buffer_t *bf=backend_buffer_alloc(g_be,sizeof(float));

    for(int pos=0;pos<steps;pos++){
        float*wte=m->token_embedding_table;
        memcpy(x,wte+token*D,D*sizeof(float));
        if(pt==BM_POS_LEARNED&&m->wpe)for(int i=0;i<D;i++)x[i]+=m->wpe[pos*D+i];

        for(int l=0;l<L;l++){
            B(); R(x,m->ln1w+l*D); C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));

            float _q[NH*HD],_k[KV],_v[KV];
            B();
            E2(bq,b,m->qw+l*NH*HD*D,0,1,D,NH*HD);
            E2(bk,b,m->kw+l*NKV*HD*D,0,1,D,KV);
            E2(bv,b,m->vw+l*NKV*HD*D,0,1,D,KV);
            C();
            memcpy(_q,backend_buffer_map(bq),NH*HD*sizeof(float));
            memcpy(_k,backend_buffer_map(bk),KV*sizeof(float));
            memcpy(_v,backend_buffer_map(bv),KV*sizeof(float));

            // QK norm (Gemma-style)
            if(m->arch.has_qk_norm){
                B();
                float* qnw=m->q_norm_w+l*NH*HD;
                memcpy(backend_buffer_map(bi),_q,NH*HD*sizeof(float));
                memcpy(backend_buffer_map(bw),qnw,NH*HD*sizeof(float));
                memcpy(backend_buffer_map(bp),(int[]){NH,HD},2*sizeof(int));
                backend_buffer_t*_qn[]={bi,bw,bo,bp,beps}; D(kr,_qn,5,NH,1,1,1,1,1);
                memcpy(_q,backend_buffer_map(bo),NH*HD*sizeof(float));
                float*knw=m->k_norm_w+l*m->n_kv_heads*HD;
                memcpy(backend_buffer_map(bi),_k,KV*sizeof(float));
                memcpy(backend_buffer_map(bw),knw,KV*sizeof(float));
                memcpy(backend_buffer_map(bp),(int[]){m->n_kv_heads,HD},2*sizeof(int));
                backend_buffer_t*_kn[]={bi,bw,bo,bp,beps}; D(kr,_kn,5,m->n_kv_heads,1,1,1,1,1);
                C();
                memcpy(_q,backend_buffer_map(bo),NH*HD*sizeof(float));
                memcpy(_k,backend_buffer_map(bo),KV*sizeof(float));
                memcpy(backend_buffer_map(bq),_q,NH*HD*sizeof(float));
                memcpy(backend_buffer_map(bk),_k,KV*sizeof(float));
            }

            // RoPE on GPU
            if(pt==BM_POS_ROPE){
                *(int*)backend_buffer_map(br)=HD;
                int ps=pos;
                *(float*)backend_buffer_map(bf)=(float)ps;
                B();
                backend_buffer_t *_rp[]={bq,bk,br,bf};
                D(kp,_rp,4,NH+m->n_kv_heads,1,1,1,1,1);
                C();
                memcpy(_q,backend_buffer_map(bq),NH*HD*sizeof(float));
                memcpy(_k,backend_buffer_map(bk),KV*sizeof(float));
            }

            // KV cache store
            float*kk=kvc+l*2*MS*KV,*kv=kk+MS*KV;
            memcpy(kk+pos*KV,_k,KV*sizeof(float));
            memcpy(kv+pos*KV,_v,KV*sizeof(float));

            // Attention: GPU matmul for QK^T and SV
            int S=pos+1;
            float*xa=calloc(NH*HD,sizeof(float));
            for(int h=0;h<NH;h++){
                int kh=h/KM;
                float*kb=calloc(S*HD,sizeof(float)),*vb=calloc(S*HD,sizeof(float));
                float*scores=calloc(S,sizeof(float)),*ho=calloc(HD,sizeof(float));
                for(int t=0;t<S;t++)for(int i=0;i<HD;i++){
                    kb[t*HD+i]=kk[kh*KV+t*KV+i]; vb[t*HD+i]=kv[kh*KV+t*KV+i];
                }
                B();
                E(_q+h*HD,kb,0,1,HD,S);
                C();
                memcpy(scores,backend_buffer_map(bo),S*sizeof(float));
                for(int t=0;t<S;t++)scores[t]*=sc;
                float mx=-INFINITY,sum=0;
                for(int t=0;t<S;t++)if(scores[t]>mx)mx=scores[t];
                for(int t=0;t<S;t++){scores[t]=expf(scores[t]-mx);sum+=scores[t];}
                for(int t=0;t<S;t++)scores[t]/=sum;
                B();
                E(scores,vb,0,1,S,HD);
                C();
                memcpy(ho,backend_buffer_map(bo),HD*sizeof(float));
                for(int i=0;i<HD;i++)xa[h*HD+i]=ho[i];
                free(kb);free(vb);free(scores);free(ho);
            }
            // Output proj + residual
            B();E(xa,m->attprojw+l*NH*HD*D,0,1,NH*HD,D);C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));
            for(int i=0;i<D;i++)x[i]+=b[i]; free(xa);
            // Post-attention norm (always applied for Gemma, same as ln2w for Llama)
            if(m->arch.has_ffn_post_norm){
                B();R(x,m->ln2w+l*D);C();
                memcpy(x,backend_buffer_map(bo),D*sizeof(float));
            }
            // FFN norm + gate + up + act + down
            B();
            if(m->arch.has_ffn_post_norm){
                R(x,m->pre_ffn_w+l*D);
            }else{
                R(x,m->ln2w+l*D);
            }
            E(backend_buffer_map(bo),m->fcw+l*H*D,0,1,D,H);
            if(at==BM_ACT_SWIGLU){E(backend_buffer_map(bo),m->fcw3+l*H*D,0,1,D,H);}
            C();
            memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
            if(at==BM_ACT_SWIGLU){
                memcpy(hb2,backend_buffer_map(bo),H*sizeof(float));
                B();S(hb,hb2,H);C();
                memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
            }else{
                B();G(hb,H);C();
                memcpy(hb,backend_buffer_map(bo),H*sizeof(float));
            }
            B();E(hb,m->fcprojw+l*D*H,0,1,H,D);C();
            memcpy(b,backend_buffer_map(bo),D*sizeof(float));
            for(int i=0;i<D;i++)x[i]+=b[i];
            if(m->arch.has_ffn_post_norm){
                B();R(x,m->ffn_post_w+l*D);C();
                memcpy(x,backend_buffer_map(bo),D*sizeof(float));
            }
        }
        // Final norm + classifier
        B();R(x,m->lnfw);E(backend_buffer_map(bo),m->wcls,0,1,D,V);C();
        memcpy(logits,backend_buffer_map(bo),V*sizeof(float));
        if (pos == 0) {
            FILE* lf = fopen("test/our_logits.bin", "wb");
            if (lf) {
                int Vv = V;
                fwrite(&Vv, sizeof(int), 1, lf);
                fwrite(logits, sizeof(float), V, lf);
                fclose(lf);
            }
        }
        // Sample
        if(pos<nt-1)next=ptok[pos+1];
        else{if(temp==0){next=0;float mv=logits[0];for(int i=1;i<V;i++)if(logits[i]>mv){mv=logits[i];next=i;}}
             else{float sum=0;for(int i=0;i<V;i++){logits[i]=expf(logits[i]/temp);sum+=logits[i];}
                  float r=(float)rand()/(float)RAND_MAX*sum,c=0;
                  for(next=0;next<V;next++){c+=logits[next];if(c>=r)break;}}}
        bm_tokenizer_safe_print(bm_tokenizer_decode(&tok,prev,next));
        prev=token;token=next;
        fprintf(stderr,".");
    }
    printf("\n");
    backend_buffer_free(bq);backend_buffer_free(bk);backend_buffer_free(bv);
    backend_buffer_free(br);backend_buffer_free(bf);
    free(ptok);bm_tokenizer_free(&tok);free(kvc);free(x);free(b);free(logits);free(hb);free(hb2);
    backend_buffer_free(bi);backend_buffer_free(bw);backend_buffer_free(bo);backend_buffer_free(bp);backend_buffer_free(beps);
    backend_kernel_destroy(km);backend_kernel_destroy(kr);backend_kernel_destroy(kl);
    backend_kernel_destroy(kg);backend_kernel_destroy(ks);backend_kernel_destroy(kp);
    return 0;
}
