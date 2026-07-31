// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/tokenizer.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((bm_token_index_t*)a)->str, ((bm_token_index_t*)b)->str);
}

static void init_byte_unicode_tables(bm_tokenizer_t* t) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            t->b2u[b] = b;
        } else {
            t->b2u[b] = 256 + n;
            n++;
        }
    }
    for (int i = 0; i < 512; i++) t->u2b[i] = 0;
    for (int b = 0; b < 256; b++) {
        t->u2b[t->b2u[b]] = (unsigned char)b;
    }
}

static int codepoint_to_utf8(int cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0';
        return 3;
    } else {
        out[0] = (char)(0xF0 | (cp >> 18));
        out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (char)(0x80 | (cp & 0x3F));
        out[4] = '\0';
        return 4;
    }
}

static int utf8_to_codepoint(const char* s, int* cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    } else if ((c & 0xE0) == 0xC0) {
        *cp = ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        return 2;
    } else if ((c & 0xF0) == 0xE0) {
        *cp = ((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
        return 3;
    } else if ((c & 0xF8) == 0xF0) {
        *cp = ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12) | (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
        return 4;
    }
    *cp = c;
    return 1;
}

static char* decode_token_b2u(bm_tokenizer_t* t, const char* token_str) {
    int raw_len = strlen(token_str);
    char* decoded = (char*)malloc(raw_len + 1);
    int dpos = 0;
    int i = 0;
    while (token_str[i] != '\0') {
        int cp;
        int adv = utf8_to_codepoint(token_str + i, &cp);
        if (cp >= 0 && cp < 512) {
            decoded[dpos++] = (char)t->u2b[cp];
        } else {
            decoded[dpos++] = (char)cp;
        }
        i += adv;
    }
    decoded[dpos] = '\0';
    return decoded;
}

// --- JSON parsing helpers for tokenizer.json ---

static int man_bsearch(char* key, bm_token_index_t* arr, int n);

static void json_skip_ws(const char** p) {
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++;
}

static int json_parse_string(const char** p, char* out, int max_len) {
    if (**p != '"') return -1;
    (*p)++;
    int len = 0;
    while (**p && **p != '"') {
        if (**p == '\\' && (*p)[1]) {
            (*p)++;
            switch (**p) {
                case '"':  out[len++] = '"';  break;
                case '\\': out[len++] = '\\'; break;
                case '/':  out[len++] = '/';  break;
                case 'n':  out[len++] = '\n';  break;
                case 't':  out[len++] = '\t';  break;
                case 'r':  out[len++] = '\r';  break;
                case 'b':  out[len++] = '\b';  break;
                case 'f':  out[len++] = '\f';  break;
                case 'u': {
                    (*p)++;
                    int cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char c = (*p)[i];
                        int d;
                        if (c >= '0' && c <= '9') d = c - '0';
                        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
                        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
                        else { out[len] = '\0'; return -1; }
                        cp = (cp << 4) | d;
                    }
                    (*p) += 3;
                    len += codepoint_to_utf8(cp, out + len);
                    break;
                }
                default: out[len++] = **p; break;
            }
            (*p)++;
        } else {
            out[len++] = **p;
            (*p)++;
        }
        if (len >= max_len - 1) break;
    }
    out[len] = '\0';
    if (**p == '"') (*p)++;
    return len;
}

static int json_parse_int(const char** p) {
    json_skip_ws(p);
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    int val = 0;
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

// --- Binary format loader (legacy tokenizer.bin) ---

static void bm_tokenizer_init_from_bin(bm_tokenizer_t* t, const char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    t->decoded_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    init_byte_unicode_tables(t);

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

    t->decoded_vocab = (char**)malloc(vocab_size * sizeof(char*));
    for (int i = 0; i < vocab_size; i++) {
        t->decoded_vocab[i] = decode_token_b2u(t, t->vocab[i]);
    }
}

// --- JSON format loader (tokenizer.json, no preprocessing needed) ---

static void bm_tokenizer_init_from_json(bm_tokenizer_t* t, const char* json_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)calloc(vocab_size, sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    t->decoded_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    init_byte_unicode_tables(t);

    for (int i = 0; i < vocab_size; i++) {
        t->vocab_scores[i] = -1e20f;
    }

    FILE* f = fopen(json_path, "rb");
    if (!f) {
        BMT_LOG_ERROR("couldn't load tokenizer: %s", json_path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json = (char*)malloc(file_size + 1);
    fread(json, 1, file_size, f);
    json[file_size] = '\0';
    fclose(f);

    t->max_token_length = 1;

    const char* p = strstr(json, "\"vocab\"");
    if (p) {
        p += 7;
        json_skip_ws(&p);
        if (*p == ':') p++;
        json_skip_ws(&p);
        if (*p == '{') {
            p++;
            while (*p) {
                json_skip_ws(&p);
                if (*p == '}') { p++; break; }
                if (*p == ',') { p++; continue; }

                char token_buf[512];
                int tlen = json_parse_string(&p, token_buf, sizeof(token_buf));
                if (tlen < 0) break;
                json_skip_ws(&p);
                if (*p == ':') p++;
                int id = json_parse_int(&p);
                if (id < 0 || id >= vocab_size) continue;

                int slen = strlen(token_buf);
                t->vocab[id] = (char*)malloc(slen + 1);
                memcpy(t->vocab[id], token_buf, slen + 1);
                if (slen > (int)t->max_token_length) t->max_token_length = slen;
            }
        }
    }

    for (int i = 0; i < vocab_size; i++) {
        if (!t->vocab[i]) {
            t->vocab[i] = (char*)malloc(1);
            t->vocab[i][0] = '\0';
        }
    }

    t->sorted_vocab = malloc(t->vocab_size * sizeof(bm_token_index_t));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(bm_token_index_t), compare_tokens);

    p = strstr(json, "\"merges\"");
    if (p) {
        p += 8;
        json_skip_ws(&p);
        if (*p == ':') p++;
        json_skip_ws(&p);
        if (*p == '[') {
            p++;
            int rank = 0;
            while (*p) {
                json_skip_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }

                char merge_buf[1024];
                int mlen = json_parse_string(&p, merge_buf, sizeof(merge_buf));
                if (mlen < 0) break;

                char* space = strchr(merge_buf, ' ');
                if (space) {
                    *space = '\0';
                    char merged[1024];
                    int p1len = strlen(merge_buf);
                    int p2len = strlen(space + 1);
                    memcpy(merged, merge_buf, p1len);
                    memcpy(merged + p1len, space + 1, p2len + 1);
                    int id = man_bsearch(merged, t->sorted_vocab, t->vocab_size);
                    if (id >= 0) {
                        t->vocab_scores[id] = -(float)rank;
                    }
                }
                rank++;
                json_skip_ws(&p);
                if (*p == ',') p++;
            }
        }
    }

    free(json);

    t->decoded_vocab = (char**)malloc(vocab_size * sizeof(char*));
    for (int i = 0; i < vocab_size; i++) {
        t->decoded_vocab[i] = decode_token_b2u(t, t->vocab[i]);
    }
}

void bm_tokenizer_init(bm_tokenizer_t* t, const char* path, int vocab_size) {
    char json_path[512];
    snprintf(json_path, sizeof(json_path), "%s/tokenizer.json", path);
    FILE* f = fopen(json_path, "rb");
    if (f) {
        fclose(f);
        bm_tokenizer_init_from_json(t, json_path, vocab_size);
        return;
    }
    bm_tokenizer_init_from_bin(t, path, vocab_size);
}

void bm_tokenizer_free(bm_tokenizer_t* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
    if (t->decoded_vocab) {
        for (int i = 0; i < t->vocab_size; i++) free(t->decoded_vocab[i]);
        free(t->decoded_vocab);
    }
}

char* bm_tokenizer_decode(bm_tokenizer_t* t, int prev_token, int token) {
    (void)prev_token;
    if (t->decoded_vocab && token >= 0 && token < t->vocab_size) {
        return t->decoded_vocab[token];
    }
    return t->vocab[token];
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
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    const unsigned char* bytes = (const unsigned char*)text;
    size_t text_len = strlen(text);
    for (size_t i = 0; i < text_len; i++) {
        unsigned char b = bytes[i];
        int cp = t->b2u[b];
        char utf8[5];
        codepoint_to_utf8(cp, utf8);
        int id = man_bsearch(utf8, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            tokens[(*n_tokens)++] = b;
        }
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
