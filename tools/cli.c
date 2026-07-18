#include "baremetal.h"
#include "baremetal/model.h"
#include "backend/backend.h"
#include "backend/metal/device.h"
#include "utils/log.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void print_usage(const char* prog) {
    printf("bare.metal - LLM inference engine for Apple Silicon\n\n");
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  run <checkpoint> [prompt]   Run inference on a model\n");
    printf("  info <checkpoint>           Print model information\n");
    printf("  test-dispatch               Test Metal kernel dispatch\n");
    printf("  test-matmul                 Test Metal matmul kernel\n");
    printf("\nOptions:\n");
    printf("  -t, --temperature <float>   Sampling temperature (default: 1.0)\n");
    printf("  -p, --topp <float>          Top-p threshold (default: 0.9)\n");
    printf("  -n, --steps <int>           Max generation steps (default: 256)\n");
    printf("  -s, --seed <int>            RNG seed (default: time-based)\n");
}

static int cmd_test_dispatch(void) {
#define TEST_N 8
    float result[TEST_N];
    backend_ctx_t* be = backend_create();
    if (!be) { fprintf(stderr, "Failed to create backend\n"); return 1; }
    if (!be->library) {
        fprintf(stderr, "No Metal library loaded\n");
        backend_destroy(be);
        return 1;
    }

    backend_kernel_t* kn = backend_kernel_create(be, "vec_add");
    if (!kn) { fprintf(stderr, "vec_add kernel not found\n"); backend_destroy(be); return 1; }

    float a[TEST_N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[TEST_N] = {10, 20, 30, 40, 50, 60, 70, 80};

    backend_buffer_t* bufA = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufB = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufR = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufN = backend_buffer_alloc(be, sizeof(int));

    memcpy(backend_buffer_map(bufA), a, TEST_N * sizeof(float));
    memcpy(backend_buffer_map(bufB), b, TEST_N * sizeof(float));
    int n_val = TEST_N;
    memcpy(backend_buffer_map(bufN), &n_val, sizeof(int));
    backend_buffer_unmap(bufA);
    backend_buffer_unmap(bufB);
    backend_buffer_unmap(bufN);

    backend_buffer_t* bufs[] = {bufA, bufB, bufR, bufN};
    int ok = 1;
    if (backend_kernel_dispatch(be, kn, bufs, NULL, 4,
                               TEST_N, 1, 1,  TEST_N, 1, 1) != 0) {
        fprintf(stderr, "Dispatch failed\n");
        ok = 0;
        goto dispatch_cleanup;
    }

    memcpy(result, backend_buffer_map(bufR), TEST_N * sizeof(float));

    printf("GPU vec_add test: ");
    for (int i = 0; i < TEST_N; i++) {
        printf("%.0f ", result[i]);
        if (result[i] != a[i] + b[i]) ok = 0;
    }
    printf("\n%s\n", ok ? "PASS" : "FAIL");

dispatch_cleanup:
    backend_buffer_free(bufA);
    backend_buffer_free(bufB);
    backend_buffer_free(bufR);
    backend_buffer_free(bufN);
    backend_kernel_destroy(kn);
    backend_destroy(be);
    return ok ? 0 : 1;
}

static int cmd_test_matmul(void) {
    backend_ctx_t* be = backend_create();
    if (!be || !be->library) { fprintf(stderr, "No Metal backend\n"); return 1; }

    backend_kernel_t* kn = backend_kernel_create(be, "matmul_forward_naive");
    if (!kn) { fprintf(stderr, "Kernel not found\n"); backend_destroy(be); return 1; }

    const int BT = 1, C = 4, OC = 3;
    float inp[4] = {1, 2, 3, 4};
    float weight[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    int params[4] = {BT, C, OC, 0}; // has_bias=0

    backend_buffer_t* buf_inp = backend_buffer_alloc(be, BT * C * sizeof(float));
    backend_buffer_t* buf_wgt = backend_buffer_alloc(be, OC * C * sizeof(float));
    backend_buffer_t* buf_bias = backend_buffer_alloc(be, sizeof(float));
    backend_buffer_t* buf_out = backend_buffer_alloc(be, BT * OC * sizeof(float));
    backend_buffer_t* buf_params = backend_buffer_alloc(be, 4 * sizeof(int));

    memcpy(backend_buffer_map(buf_inp), inp, BT * C * sizeof(float));
    memcpy(backend_buffer_map(buf_wgt), weight, OC * C * sizeof(float));
    memcpy(backend_buffer_map(buf_params), params, 4 * sizeof(int));

    backend_buffer_t* bufs[] = {buf_inp, buf_wgt, buf_bias, buf_out, buf_params};
    int ok = 1;
    if (backend_kernel_dispatch(be, kn, bufs, NULL, 5,
                               BT, OC, 1,  1, 1, 1) != 0) {
        fprintf(stderr, "Dispatch failed\n");
        ok = 0;
        goto matmul_cleanup;
    }

    float out[3];
    memcpy(out, backend_buffer_map(buf_out), BT * OC * sizeof(float));
    printf("GPU matmul test: [%.0f %.0f %.0f] ",
           out[0], out[1], out[2]);
    if (out[0]==1 && out[1]==2 && out[2]==3) {
        printf("PASS\n");
    } else {
        printf("FAIL (expected [1 2 3])\n");
        ok = 0;
    }

matmul_cleanup:
    backend_buffer_free(buf_inp);
    backend_buffer_free(buf_wgt);
    backend_buffer_free(buf_bias);
    backend_buffer_free(buf_out);
    backend_buffer_free(buf_params);
    backend_kernel_destroy(kn);
    backend_destroy(be);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    const char* cmd = argv[1];

    if (strcmp(cmd, "test-dispatch") == 0) return cmd_test_dispatch();
    if (strcmp(cmd, "test-matmul") == 0) return cmd_test_matmul();

    if (strcmp(cmd, "info") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
        bm_model_t* model = calloc(1, sizeof(*model));
        bm_load_weights(model, argv[2]);
        bm_print_model_info(model);
        bm_destroy_model(model);
        bm_destroy(ctx);
        return 0;
    }

    if (strcmp(cmd, "run") == 0) {
        printf("Inference mode — not yet implemented.\n");
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
