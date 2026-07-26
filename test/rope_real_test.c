#include "backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static float* loadf(const char*d,const char*n,size_t c){char p[256];snprintf(p,256,"%s/%s.bin",d,n);float*b=malloc(c*4);FILE*f=fopen(p,"rb");fread(b,4,c,f);fclose(f);return b;}
static double cosmax(const float*a,const float*b,size_t n,float*m){double dot=0,na=0,nb=0;*m=0;for(size_t i=0;i<n;i++){float x=a[i],y=b[i];dot+=x*y;na+=x*x;nb+=y*y;float dd=fabsf(x-y);if(dd>*m)*m=dd;}return dot/(sqrt(na)*sqrt(nb)+1e-12);}
int main(void){
  const int NH=9,NKV=3,HD=64,S=4;const float theta=100000.0f;const char*G="test/grad";
  backend_ctx_t*be=backend_create();
  float*gq=loadf(G,"roreal_gout_q",S*NH*HD),*gk=loadf(G,"roreal_gout_k",S*NKV*HD),*eQ=loadf(G,"roreal_gQ",S*NH*HD),*eK=loadf(G,"roreal_gK",S*NKV*HD);
  backend_buffer_t*b1=backend_buffer_alloc(be,S*NH*HD*4),*b2=backend_buffer_alloc(be,S*NKV*HD*4),*o1=backend_buffer_alloc(be,S*NH*HD*4),*o2=backend_buffer_alloc(be,S*NKV*HD*4),*bp=backend_buffer_alloc(be,16*4),*bt=backend_buffer_alloc(be,4);
  memcpy(backend_buffer_map(b1),gq,S*NH*HD*4);backend_buffer_unmap(b1);memcpy(backend_buffer_map(b2),gk,S*NKV*HD*4);backend_buffer_unmap(b2);
  int par[4]={HD,NKV,S,NH};memcpy(backend_buffer_map(bp),par,16);backend_buffer_unmap(bp);memcpy(backend_buffer_map(bt),&theta,4);backend_buffer_unmap(bt);
  backend_kernel_t*k=backend_kernel_create(be,"rope_backward_seq");
  {backend_encoder_t*enc=backend_encode_begin(be);backend_buffer_t*bs[]={b1,b2,o1,o2,bp,bt};backend_encode_dispatch(enc,k,bs,NULL,6,S*NH,1,1,S*NH,1,1);backend_encode_commit(enc);backend_encode_wait(enc);}
  float*O1=backend_buffer_map(o1),*O2=backend_buffer_map(o2);float m;
  printf("gQ cos=%.6f\n",cosmax(O1,eQ,S*NH*HD,&m));printf("gK cos=%.6f\n",cosmax(O2,eK,S*NKV*HD,&m));
  int ok=cosmax(O1,eQ,S*NH*HD,&m)>0.9999&&cosmax(O2,eK,S*NKV*HD,&m)>0.9999;
  printf("\n=== rope_seq real-size: %s ===\n",ok?"PASS":"FAIL");
  backend_buffer_unmap(o1);backend_buffer_unmap(o2);free(gq);free(gk);free(eQ);free(eK);
  backend_buffer_free(b1);backend_buffer_free(b2);backend_buffer_free(o1);backend_buffer_free(o2);backend_buffer_free(bp);backend_buffer_free(bt);backend_kernel_destroy(k);backend_destroy(be);return ok?0:1;}
