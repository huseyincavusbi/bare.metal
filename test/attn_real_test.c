#include "backend/backend.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static float* loadf(const char* d, const char* n, size_t c){char p[256];snprintf(p,256,"%s/%s.bin",d,n);float*b=malloc(c*4);FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(1);}fread(b,4,c,f);fclose(f);return b;}
static double cosmax(const float*a,const float*b,size_t n,float*m){double dot=0,na=0,nb=0;*m=0;for(size_t i=0;i<n;i++){float x=a[i],y=b[i];dot+=x*y;na+=x*x;nb+=y*y;float dd=fabsf(x-y);if(dd>*m)*m=dd;}return dot/(sqrt(na)*sqrt(nb)+1e-12);}
int main(void){
  const int NH=9,NKV=3,HD=64,S=4,kv_mul=3;const float scale=0.125f;
  const char*G="test/grad";
  backend_ctx_t*be=backend_create();if(!be){fprintf(stderr,"no be\n");return 1;}
  float*Q=loadf(G,"attn_real_Q",S*NH*HD),*K=loadf(G,"attn_real_K",S*NKV*HD),*V=loadf(G,"attn_real_V",S*NKV*HD);
  float*gout=loadf(G,"attn_real_gout",S*NH*HD),*eQ=loadf(G,"attn_real_gQ",S*NH*HD),*eK=loadf(G,"attn_real_gK",S*NKV*HD),*eV=loadf(G,"attn_real_gV",S*NKV*HD);
  backend_buffer_t*bQ=backend_buffer_alloc(be,S*NH*HD*4),*bK=backend_buffer_alloc(be,S*NKV*HD*4),*bV=backend_buffer_alloc(be,S*NKV*HD*4),*bO=backend_buffer_alloc(be,S*NH*HD*4),*bP=backend_buffer_alloc(be,16*4),*bS=backend_buffer_alloc(be,4);
  backend_buffer_t*gQ=backend_buffer_alloc(be,S*NH*HD*4),*gK=backend_buffer_alloc(be,S*NKV*HD*4),*gV=backend_buffer_alloc(be,S*NKV*HD*4),*bG=backend_buffer_alloc(be,S*NH*HD*4);
  memcpy(backend_buffer_map(bQ),Q,S*NH*HD*4);backend_buffer_unmap(bQ);memcpy(backend_buffer_map(bK),K,S*NKV*HD*4);backend_buffer_unmap(bK);memcpy(backend_buffer_map(bV),V,S*NKV*HD*4);backend_buffer_unmap(bV);memcpy(backend_buffer_map(bS),&scale,4);backend_buffer_unmap(bS);
  int par[5]={NH,S,HD,NKV,kv_mul};memcpy(backend_buffer_map(bP),par,20);backend_buffer_unmap(bP);
  memset(backend_buffer_map(gK),0,S*NKV*HD*4);backend_buffer_unmap(gK);memset(backend_buffer_map(gV),0,S*NKV*HD*4);backend_buffer_unmap(gV);
  backend_kernel_t*kf=backend_kernel_create(be,"attention_forward_seq");
  {backend_encoder_t*enc=backend_encode_begin(be);backend_buffer_t*bs[]={bQ,bK,bV,bO,bP,bS};backend_encode_dispatch(enc,kf,bs,NULL,6,S*NH,1,1,S*NH,1,1);backend_encode_commit(enc);backend_encode_wait(enc);}
  memcpy(backend_buffer_map(bK),K,S*NKV*HD*4);backend_buffer_unmap(bK);memcpy(backend_buffer_map(bV),V,S*NKV*HD*4);backend_buffer_unmap(bV);memcpy(backend_buffer_map(bQ),Q,S*NH*HD*4);backend_buffer_unmap(bQ);
  memcpy(backend_buffer_map(bG),gout,S*NH*HD*4);backend_buffer_unmap(bG);
  backend_kernel_t*kb=backend_kernel_create(be,"attention_backward");
  {backend_encoder_t*enc=backend_encode_begin(be);backend_buffer_t*bs[]={bG,bQ,bK,bV,gQ,gK,gV,bP,bS};backend_encode_dispatch(enc,kb,bs,NULL,9,S*NH,1,1,S*NH,1,1);backend_encode_commit(enc);backend_encode_wait(enc);}
  float*oQ=backend_buffer_map(gQ),*oK=backend_buffer_map(gK),*oV=backend_buffer_map(gV);
  float m;printf("gQ cos=%.6f\n",cosmax(oQ,eQ,S*NH*HD,&m));printf("gK cos=%.6f\n",cosmax(oK,eK,S*NKV*HD,&m));printf("gV cos=%.6f\n",cosmax(oV,eV,S*NKV*HD,&m));
  int ok=cosmax(oQ,eQ,S*NH*HD,&m)>0.9999&&cosmax(oK,eK,S*NKV*HD,&m)>0.9999&&cosmax(oV,eV,S*NKV*HD,&m)>0.9999;
  printf("\n=== attn real-size: %s ===\n",ok?"PASS":"FAIL");
  backend_buffer_unmap(gQ);backend_buffer_unmap(gK);backend_buffer_unmap(gV);
  free(Q);free(K);free(V);free(gout);free(eQ);free(eK);free(eV);
  backend_buffer_free(bQ);backend_buffer_free(bK);backend_buffer_free(bV);backend_buffer_free(bO);backend_buffer_free(bP);backend_buffer_free(bS);backend_buffer_free(gQ);backend_buffer_free(gK);backend_buffer_free(gV);backend_buffer_free(bG);
  backend_kernel_destroy(kf);backend_kernel_destroy(kb);backend_destroy(be);return ok?0:1;}
