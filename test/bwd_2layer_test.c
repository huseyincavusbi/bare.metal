/* bwd_2layer_test.c -- bisect: 2-layer full backward vs PyTorch 2-layer model. */
#include "baremetal.h"
#include "baremetal/model.h"
#include "baremetal/graph.h"
#include "baremetal/scheduler.h"
#include "kernels/registry.h"
#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static float* loadf(const char* d, const char* n, size_t c){char p[256];snprintf(p,256,"%s/%s.bin",d,n);float*b=malloc(c*4);FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(1);}fread(b,4,c,f);fclose(f);return b;}
static double cosmax(const float*a,const float*b,size_t n,float*m){double dot=0,na=0,nb=0;*m=0;for(size_t i=0;i<n;i++){float x=a[i],y=b[i];dot+=x*y;na+=x*x;nb+=y*y;float dd=fabsf(x-y);if(dd>*m)*m=dd;}return dot/(sqrt(na)*sqrt(nb)+1e-12);}
static int find_w(bmt_graph_t*g,void*ptr){for(int i=0;i<g->n_tensors;i++)if(g->tensors[i].type==BMT_TENSOR_TYPE_WEIGHT&&g->tensors[i].weight_ptr==ptr)return i;return -1;}
int main(void){
  const char*md="data/smollm2-135m";const int tokens[]={6403,1980,253,655},targets_arr[]={1980,253,655,0},S=4;const char*G="test/grad";
  backend_ctx_t*be=backend_create();if(!be){fprintf(stderr,"no be\n");return 1;}
  bm_model_t*m=calloc(1,sizeof(bm_model_t));bm_load_weights(m,md);
  m->arch.n_layers=2;   /* OVERRIDE: only 2 layers */
  int D=m->arch.dim,V=m->arch.vocab_size,NH=m->arch.n_heads,HD=m->head_size,NKV=m->n_kv_heads,H=m->arch.hidden_dim;
  bmt_graph_build_train(m,S);bmt_graph_t*g=(bmt_graph_t*)m->graph;
  bmk_registry_t*reg=bmk_registry_create(be);
  bmk_register(reg,0,0,"matmul_forward_naive");bmk_register(reg,1,0,"rmsnorm_forward");bmk_register(reg,4,0,"swiglu_forward");
  bmt_scheduler_t*sched=bmt_scheduler_create(be,reg,g,S,m->kv_dim,2);
  float*x=malloc((size_t)S*D*4);for(int s=0;s<S;s++)memcpy(x+s*D,m->token_embedding_table+tokens[s]*D,D*4);
  bmt_scheduler_set_input(sched,0,x,(size_t)S*D*4);bmt_scheduler_forward_train(sched,S);
  int t_logits=g->n_tensors-1;
  float loss=bmt_scheduler_xent_backward(sched,t_logits,targets_arr,S,V);
  printf("[2-layer] our loss=%.4f (ref 16.1266)\n",loss);
  bmt_scheduler_backward(sched);
  int fails=0;
  struct{const char*nm;void*ptr;size_t n;}sm[]={
    {"b2_l0_qw",m->qw+0*NH*HD*D,(size_t)NH*HD*D},{"b2_l0_kw",m->kw+0*NKV*HD*D,(size_t)NKV*HD*D},
    {"b2_l0_vw",m->vw+0*NKV*HD*D,(size_t)NKV*HD*D},{"b2_l0_ow",m->attprojw+0*NH*HD*D,(size_t)NH*HD*D},
    {"b2_l0_ln1w",m->ln1w+0*D,(size_t)D},{"b2_l0_ln2w",m->ln2w+0*D,(size_t)D},{"b2_l0_dw",m->fcprojw+0*D*H,(size_t)D*H},
    {"b2_l1_qw",m->qw+1*NH*HD*D,(size_t)NH*HD*D},{"b2_l1_ow",m->attprojw+1*NH*HD*D,(size_t)NH*HD*D},{"b2_l1_dw",m->fcprojw+1*D*H,(size_t)D*H},
    {"b2_lnfw",m->lnfw,(size_t)D},
  };
  int ns=sizeof(sm)/sizeof(sm[0]);
  for(int i=0;i<ns;i++){int tid=find_w(g,sm[i].ptr);if(tid<0){printf("FAIL %-11s not found\n",sm[i].nm);fails++;continue;}
    float*our=malloc(sm[i].n*4);bmt_scheduler_get_grad(sched,tid,our,sm[i].n*4);float*ref=loadf(G,sm[i].nm,sm[i].n);
    float mx;double c=cosmax(our,ref,sm[i].n,&mx);printf("%-11s cos=%.6f maxabs=%.2e %s\n",sm[i].nm,c,mx,c>0.9999?"OK":"BAD");if(c<=0.9999)fails++;free(our);free(ref);}
  int ok=fails==0;printf("\n=== 2-layer backward: %s (%d/%d) ===\n",ok?"PASS":"FAIL",ns-fails,ns);
  free(x);bmt_scheduler_destroy(sched);bmk_registry_destroy(reg);bm_destroy_model(m);backend_destroy(be);return ok?0:1;}
