#include "baremetal.h"
#include "backend/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "Failed to create backend\n"); return 1; }
    
    backend_kernel_t* km = backend_kernel_create(be, "matmul_forward_naive");
    if (!km) { fprintf(stderr, "Kernel not found\n"); return 1; }
    
    // Allocate fresh buffers
    backend_buffer_t* bi = backend_buffer_alloc(be, 576 * sizeof(float));
    backend_buffer_t* bw = backend_buffer_alloc(be, 576 * 576 * sizeof(float));
    backend_buffer_t* bo = backend_buffer_alloc(be, 576 * sizeof(float));
    backend_buffer_t* bp = backend_buffer_alloc(be, 4 * sizeof(int));
    backend_buffer_t* bbs = backend_buffer_alloc(be, 576 * sizeof(float));
    
    // Load test data
    FILE* f = fopen("debug/l0_matmul_bi.bin", "rb");
    if (!f) { fprintf(stderr, "Failed to open input\n"); return 1; }
    float inp[576];
    fread(inp, sizeof(float), 576, f);
    fclose(f);
    
    f = fopen("debug/l0_matmul_bw.bin", "rb");
    if (!f) { fprintf(stderr, "Failed to open weight\n"); return 1; }
    float wgt[576*576];
    fread(wgt, sizeof(float), 576*576, f);
    fclose(f);
    
    // Copy to buffers
    memcpy(backend_buffer_map(bi), inp, 576 * sizeof(float));
    memcpy(backend_buffer_map(bw), wgt, 576 * 576 * sizeof(float));
    int params[] = {1, 576, 576, 0};
    memcpy(backend_buffer_map(bp), params, 4 * sizeof(int));
    
    // Dispatch
    backend_encoder_t* enc = backend_encode_begin(be);
    backend_buffer_t* bufs[] = {bi, bw, bbs, bo, bp};
    backend_encode_dispatch(enc, km, bufs, NULL, 5, 1, 576, 1, 1, 1, 1);
    backend_encode_commit(enc);
    backend_encode_wait(enc);
    
    // Read result
    float result[576];
    memcpy(result, backend_buffer_map(bo), 576 * sizeof(float));
    
    printf("Result[:5]: %.8f %.8f %.8f %.8f %.8f\n", 
           result[0], result[1], result[2], result[3], result[4]);
    
    // Cleanup
    backend_buffer_free(bi);
    backend_buffer_free(bw);
    backend_buffer_free(bo);
    backend_buffer_free(bp);
    backend_buffer_free(bbs);
    backend_kernel_destroy(km);
    backend_destroy(be);
    
    return 0;
}
