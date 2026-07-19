#include "baremetal/tokenizer.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((bm_token_index_t*)a)->str, ((bm_token_index_t*)b)->str);
}

void bm_tokenizer_init(bm_tokenizer_t* t, const char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE* file = fopen(path, "rb");
    if (!file) {
        BMT_LOG_ERROR("couldn't load tokenizer: %s", path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        BMT_LOG_ERROR("failed to read tokenizer header");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            BMT_LOG_ERROR("failed to read tokenizer score");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            BMT_LOG_ERROR("failed to read tokenizer length");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            BMT_LOG_ERROR("failed to read tokenizer string");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void bm_tokenizer_free(bm_tokenizer_t* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* bm_tokenizer_decode(bm_tokenizer_t* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

void bm_tokenizer_safe_print(char* piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    printf("%s", piece);
    fflush(stdout);
}

int bm_tokenizer_encode(bm_tokenizer_t* t, const char* text, int8_t bos,
                        int8_t eos, int* tokens, int* n_tokens) {
    if (!text) return -1;

    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;

    for (const char* c = text; *c != '\0'; c++)
        tokens[(*n_tokens)++] = (unsigned char)*c + 3;

    if (eos) tokens[(*n_tokens)++] = 2;
    return 0;
}
