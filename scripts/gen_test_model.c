#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

int main() {
    int dim = 64, hidden_dim = 172, n_layers = 2, n_heads = 4, n_kv_heads = 4;
    int vocab_size = 32000, max_seq_len = 256;
    int head_size = dim / n_heads;

    size_t L = n_layers, D = dim, H = hidden_dim, V = vocab_size, NH = n_heads, HD = head_size;

    size_t total = 0;
    total += V * D;                    // wte
    total += L * D;                    // ln1w (RMSNorm)
    total += L * 3 * D * NH * HD;     // qkvw
    total += L * NH * HD * D;         // attprojw
    total += L * D;                    // ln2w (RMSNorm)
    total += L * H * D;                // w1 (gate)
    total += L * H * D;                // w3 (up)
    total += L * D * H;                // w2 (down)
    total += D;                         // lnfw
    if (1) total += V * D;             // wcls (no weight tie)

    int32_t header[256] = {0};
    header[0] = 20250718;
    header[1] = 1;
    header[2] = 0; // fp32
    header[3] = dim;
    header[4] = hidden_dim;
    header[5] = n_layers;
    header[6] = n_heads;
    header[7] = n_kv_heads;
    header[8] = vocab_size;
    header[9] = max_seq_len;
    header[10] = 1; // RMSNorm
    header[11] = 1; // SwiGLU
    header[12] = 1; // RoPE
    header[13] = 0; // MHA
    header[14] = 0; // no bias
    header[15] = 0; // no weight tie

    FILE* f = fopen("test_model.bin", "wb");
    fwrite(header, sizeof(int32_t), 256, f);

    float* weights = calloc(total, sizeof(float));
    for (size_t i = 0; i < total; i++)
        weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;

    fwrite(weights, sizeof(float), total, f);
    fclose(f);
    free(weights);

    printf("Wrote test_model.bin: %zu params (dim=%d, layers=%d, vocab=%d)\n",
           total, dim, n_layers, vocab_size);
    return 0;
}
