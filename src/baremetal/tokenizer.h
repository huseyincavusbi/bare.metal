#ifndef BMT_TOKENIZER_H
#define BMT_TOKENIZER_H

#include <stdint.h>

typedef struct {
    char* str;
    int id;
} bm_token_index_t;

typedef struct bm_tokenizer_s {
    char**            vocab;
    float*            vocab_scores;
    bm_token_index_t* sorted_vocab;
    char**            decoded_vocab;
    int               vocab_size;
    unsigned int      max_token_length;
    unsigned char     byte_pieces[512];
    int               b2u[256];
    unsigned char     u2b[512];
} bm_tokenizer_t;

void bm_tokenizer_init(bm_tokenizer_t* t, const char* path, int vocab_size);
void bm_tokenizer_free(bm_tokenizer_t* t);
char* bm_tokenizer_decode(bm_tokenizer_t* t, int prev_token, int token);
void bm_tokenizer_safe_print(char* piece);
int bm_tokenizer_encode(bm_tokenizer_t* t, const char* text, int8_t bos,
                        int8_t eos, int* tokens, int* n_tokens);

#endif
