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
    if (!be) {
        fprintf(stderr, "Failed to create backend\n");
        return 1;
    }
    if (!be->library) {
        fprintf(stderr, "No Metal library loaded — need built .metallib\n");
        backend_destroy(be);
        return 1;
    }

    backend_kernel_t* kn = backend_kernel_create(be, "vec_add");
    if (!kn) {
        fprintf(stderr, "vec_add kernel not found in .metallib\n");
        backend_destroy(be);
        return 1;
    }

    float a[TEST_N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[TEST_N] = {10, 20, 30, 40, 50, 60, 70, 80};

    backend_buffer_t* bufA = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufB = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufR = backend_buffer_alloc(be, TEST_N * sizeof(float));
    backend_buffer_t* bufN = backend_buffer_alloc(be, sizeof(int));
    int n_val = TEST_N;

    memcpy(backend_buffer_map(bufA), a, TEST_N * sizeof(float));
    memcpy(backend_buffer_map(bufB), b, TEST_N * sizeof(float));
    memcpy(backend_buffer_map(bufN), &n_val, sizeof(int));
    backend_buffer_unmap(bufA);
    backend_buffer_unmap(bufB);
    backend_buffer_unmap(bufN);

    backend_buffer_t* bufs[] = {bufA, bufB, bufR, bufN};
    size_t offsets[] = {0, 0, 0, 0};

    int tg = 1;
    int tpt = TEST_N;
    int ok = 1;

    if (backend_kernel_dispatch(be, kn, bufs, offsets, 4, tg, tpt) != 0) {
        fprintf(stderr, "Dispatch failed\n");
        goto cleanup;
    }

    memcpy(result, backend_buffer_map(bufR), TEST_N * sizeof(float));

    printf("GPU vec_add test: ");
    for (int i = 0; i < TEST_N; i++) {
        float expected = a[i] + b[i];
        printf("%.0f ", result[i]);
        if (result[i] != expected) ok = 0;
    }
    printf("\n%s\n", ok ? "PASS" : "FAIL");

cleanup:
    backend_buffer_free(bufA);
    backend_buffer_free(bufB);
    backend_buffer_free(bufR);
    backend_buffer_free(bufN);
    backend_kernel_destroy(kn);
    backend_destroy(be);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "test-dispatch") == 0) {
        return cmd_test_dispatch();
    }

    if (strcmp(cmd, "info") == 0) {
        if (argc < 3) {
            print_usage(argv[0]);
            return 1;
        }

        bm_context_t* ctx = bm_create(BM_DEVICE_METAL);
        if (!ctx) {
            fprintf(stderr, "Failed to create context\n");
            return 1;
        }

        bm_model_t* model = calloc(1, sizeof(*model));
        if (!model) {
            bm_destroy(ctx);
            return 1;
        }

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
