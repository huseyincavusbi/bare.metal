#include "baremetal.h"
#include "baremetal/model.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void print_usage(const char* prog) {
    printf("bare.metal - LLM inference engine for Apple Silicon\n\n");
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  run <checkpoint> [prompt]   Run inference on a model\n");
    printf("  info <checkpoint>           Print model information\n");
    printf("\nOptions:\n");
    printf("  -t, --temperature <float>   Sampling temperature (default: 1.0)\n");
    printf("  -p, --topp <float>          Top-p threshold (default: 0.9)\n");
    printf("  -n, --steps <int>           Max generation steps (default: 256)\n");
    printf("  -s, --seed <int>            RNG seed (default: time-based)\n");
    printf("\nExamples:\n");
    printf("  %s info stories15M.bin\n", prog);
    printf("  %s run stories15M.bin \"Once upon a time\"\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* cmd = argv[1];

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
