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

    t->sorted_vocab = malloc(t->vocab_size * sizeof(bm_token_index_t));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(bm_token_index_t), compare_tokens);
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

static int man_bsearch(char* key, bm_token_index_t* arr, int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int cmp = strcmp(key, arr[mid].str);
        if (cmp == 0) return arr[mid].id;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

int bm_tokenizer_encode(bm_tokenizer_t* t, const char* text, int8_t bos,
                        int8_t eos, int* tokens, int* n_tokens) {
    if (!text || !t->sorted_vocab) return -1;

    char* str_buffer = malloc((t->max_token_length * 2 + 3) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0' && text[0] != ' ') {
        int dp = man_bsearch(" ", t->sorted_vocab, t->vocab_size);
        if (dp >= 0) tokens[(*n_tokens)++] = dp;
    }

    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;

        int id = man_bsearch(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                unsigned char b = (unsigned char)str_buffer[i];
                char byte_buf[5] = {0};
                int blen = 0;
                if (b < 0x21) {
                    byte_buf[0] = 0xC0 | ((0x100 + b) >> 6);
                    byte_buf[1] = 0x80 | ((0x100 + b) & 0x3F);
                    blen = 2;
                } else {
                    byte_buf[0] = b;
                    blen = 1;
                }
                int bid = man_bsearch(byte_buf, t->sorted_vocab, t->vocab_size);
                if (bid >= 0) {
                    tokens[(*n_tokens)++] = bid;
                } else {
                    tokens[(*n_tokens)++] = b + 3;
                }
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, t->max_token_length * 4 + 3, "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = man_bsearch(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
    return 0;
}
