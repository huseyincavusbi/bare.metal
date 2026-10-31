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

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int utf8_sequence_len(const unsigned char* s) {
    if (s[0] < 0x80) return 1;
    if ((s[0] & 0xE0) == 0xC0) return 2;
    if ((s[0] & 0xF0) == 0xE0) return 3;
    if ((s[0] & 0xF8) == 0xF0) return 4;
    return 1;
}

static int byte_fallback_byte(const char* token, unsigned* out) {
    if (strlen(token) != 6 || token[0] != '<' || token[1] != '0' ||
        token[2] != 'x' || token[5] != '>')
        return -1;
    int hi = hexval(token[3]);
    int lo = hexval(token[4]);
    if (hi < 0 || lo < 0) return -1;
    *out = (unsigned)((hi << 4) | lo);
    return 0;
}

static char* decode_token_spm(const char* token) {
    unsigned byte;
    if (byte_fallback_byte(token, &byte) == 0) {
        char* out = (char*)malloc(2);
        out[0] = (char)byte;
        out[1] = '\0';
        return out;
    }

    size_t len = strlen(token);
    char* out = (char*)malloc(len + 1);
    size_t d = 0;
    for (size_t i = 0; i < len;) {
        if (i + 3 <= len &&
            (unsigned char)token[i] == 0xE2 &&
            (unsigned char)token[i + 1] == 0x96 &&
            (unsigned char)token[i + 2] == 0x81) {
            out[d++] = ' ';
            i += 3;
        } else {
            out[d++] = token[i++];
        }
    }
    out[d] = '\0';
    return out;
}

static void append_metaspace(char** d) {
    *(*d)++ = (char)0xE2;
    *(*d)++ = (char)0x96;
    *(*d)++ = (char)0x81;
}

static int is_json_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static char* spm_normalize_bpe(const char* text, int prepend) {
    size_t n = strlen(text);
    char* out = (char*)malloc(n * 3 + 4);
    if (!out) return NULL;
    char* d = out;
    if (prepend && n > 0 && text[0] != ' ') append_metaspace(&d);
    for (size_t i = 0; i < n; i++) {
        if (text[i] == ' ') append_metaspace(&d);
        else *d++ = text[i];
    }
    *d = '\0';
    return out;
}

static char* spm_normalize_unigram(const char* text, int prepend, int split_words) {
    size_t n = strlen(text);
    char* out = (char*)malloc(n * 3 + 4);
    if (!out) return NULL;
    char* d = out;

    if (!split_words) {
        if (prepend && n > 0 && text[0] != ' ') append_metaspace(&d);
        for (size_t i = 0; i < n; i++) {
            if (text[i] == ' ') append_metaspace(&d);
            else *d++ = text[i];
        }
        *d = '\0';
        return out;
    }

    size_t i = 0;
    while (i < n) {
        while (i < n && is_json_space(text[i])) i++;
        if (i >= n) break;
        append_metaspace(&d);
        while (i < n && !is_json_space(text[i])) *d++ = text[i++];
    }
    *d = '\0';
    return out;
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

static int json_parse_float(const char** p, float* out) {
    json_skip_ws(p);
    char* end = NULL;
    float val = strtof(*p, &end);
    if (end == *p) return -1;
    *p = end;
    *out = val;
    return 0;
}

// --- Binary format loader (legacy tokenizer.bin) ---

static void bm_tokenizer_init_from_bin(bm_tokenizer_t* t, const char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    t->decoded_vocab = NULL;
    t->kind = BM_TOKENIZER_BPE;
    t->byte_fallback = 0;
    t->unk_id = 0;
    t->bos_id = 1;
    t->eos_id = 2;
    t->prepend_space = 0;
    t->split_words = 0;
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
    t->kind = BM_TOKENIZER_BPE;
    t->byte_fallback = 0;
    t->unk_id = 0;
    t->bos_id = 1;
    t->eos_id = 2;
    t->prepend_space = 0;
    t->split_words = 0;

    const char* model = strstr(json, "\"model\"");
    if (!model) model = json;

    int is_unigram = 0;
    const char* type_p = strstr(model, "\"type\"");
    if (type_p) {
        const char* q = strchr(type_p, ':');
        if (q) {
            q++;
            json_skip_ws(&q);
            char type_buf[32];
            if (json_parse_string(&q, type_buf, sizeof(type_buf)) > 0 &&
                strcmp(type_buf, "Unigram") == 0)
                is_unigram = 1;
        }
    }

    const char* bf = strstr(model, "\"byte_fallback\"");
    if (bf) {
        bf = strchr(bf, ':');
        if (bf) {
            bf++;
            while (*bf == ' ' || *bf == '\t' || *bf == '\n' || *bf == '\r') bf++;
            t->byte_fallback = strncmp(bf, "true", 4) == 0;
        }
    }

    const char* ui = strstr(model, "\"unk_id\"");
    if (ui) {
        ui = strchr(ui, ':');
        if (ui) t->unk_id = (int)strtol(ui + 1, NULL, 10);
    }

    int prepend_space = 0;
    int split_words = 0;
    const char* norm_p = strstr(json, "\"normalizer\"");
    if (norm_p && strstr(norm_p, "\"Prepend\"")) prepend_space = 1;
    const char* pre_p = strstr(json, "\"pre_tokenizer\"");
    if (pre_p && strstr(pre_p, "\"Metaspace\"")) {
        prepend_space = 1;
        const char* sp = strstr(pre_p, "\"split\"");
        if (sp) {
            sp = strchr(sp, ':');
            if (sp) {
                sp++;
                while (*sp == ' ' || *sp == '\t' || *sp == '\n' || *sp == '\r') sp++;
                split_words = strncmp(sp, "true", 4) == 0;
            }
        } else {
            split_words = 1;
        }
    }

    int spm_seen = 0;

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
                if (strstr(token_buf, "\xE2\x96\x81")) spm_seen = 1;
            }
        } else if (*p == '[') {
            p++;
            int id = 0;
            while (*p) {
                json_skip_ws(&p);
                if (*p == ']') { p++; break; }
                if (*p == ',') { p++; continue; }
                if (*p != '[') break;
                p++;
                json_skip_ws(&p);

                char token_buf[512];
                int tlen = json_parse_string(&p, token_buf, sizeof(token_buf));
                if (tlen < 0) break;
                json_skip_ws(&p);
                if (*p == ',') p++;
                float score = 0.0f;
                json_parse_float(&p, &score);
                json_skip_ws(&p);
                if (*p == ']') p++;

                if (id >= 0 && id < vocab_size) {
                    int slen = strlen(token_buf);
                    t->vocab[id] = (char*)malloc(slen + 1);
                    memcpy(t->vocab[id], token_buf, slen + 1);
                    t->vocab_scores[id] = score;
                    if (slen > (int)t->max_token_length) t->max_token_length = slen;
                    if (strstr(token_buf, "\xE2\x96\x81")) spm_seen = 1;
                }
                id++;
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

    if (is_unigram) t->kind = BM_TOKENIZER_UNIGRAM;
    else if (spm_seen) t->kind = BM_TOKENIZER_SPM_BPE;
    else t->kind = BM_TOKENIZER_BPE;

    t->prepend_space = prepend_space;
    t->split_words = split_words;
    if (t->unk_id == 0) {
        int uid = man_bsearch("<unk>", t->sorted_vocab, t->vocab_size);
        if (uid >= 0) t->unk_id = uid;
    }
    int sid = man_bsearch("<s>", t->sorted_vocab, t->vocab_size);
    if (sid < 0) sid = man_bsearch("<bos>", t->sorted_vocab, t->vocab_size);
    t->bos_id = sid >= 0 ? sid : -1;
    int eid = man_bsearch("</s>", t->sorted_vocab, t->vocab_size);
    if (eid < 0) eid = man_bsearch("<eos>", t->sorted_vocab, t->vocab_size);
    t->eos_id = eid >= 0 ? eid : 2;

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

                char left[512] = "";
                char right[512] = "";
                if (*p == '[') {
                    p++;
                    json_skip_ws(&p);
                    if (json_parse_string(&p, left, sizeof(left)) < 0) break;
                    json_skip_ws(&p);
                    if (*p == ',') p++;
                    json_skip_ws(&p);
                    if (json_parse_string(&p, right, sizeof(right)) < 0) break;
                    json_skip_ws(&p);
                    if (*p == ']') p++;
                } else {
                    char merge_buf[1024];
                    int mlen = json_parse_string(&p, merge_buf, sizeof(merge_buf));
                    if (mlen < 0) break;
                    char* space = strchr(merge_buf, ' ');
                    if (space) {
                        *space = '\0';
                        snprintf(left, sizeof(left), "%s", merge_buf);
                        snprintf(right, sizeof(right), "%s", space + 1);
                    }
                }

                if (left[0] && right[0]) {
                    char merged[1024];
                    snprintf(merged, sizeof(merged), "%s%s", left, right);
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
        t->decoded_vocab[i] = (t->kind == BM_TOKENIZER_BPE)
                              ? decode_token_b2u(t, t->vocab[i])
                              : decode_token_spm(t->vocab[i]);
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
    if (t->decoded_vocab && token >= 0 && token < t->vocab_size) {
        char* piece = t->decoded_vocab[token];
        if (t->kind != BM_TOKENIZER_BPE && prev_token == 1 && piece[0] == ' ')
            return piece + 1;
        return piece;
    }
    return t->vocab[token];
}

void bm_tokenizer_safe_print(char* piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (byte_val < 0x20 && byte_val != '\n' && byte_val != '\t' && byte_val != '\r')
            return;
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

static int bpe_merge(bm_tokenizer_t* t, int* tokens, int* n_tokens) {
    size_t cap = (size_t)t->max_token_length * 2 + 3;
    char* str_buffer = malloc(cap);
    if (!str_buffer) return -1;

    while (*n_tokens > 1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            int n = snprintf(str_buffer, cap, "%s%s",
                             t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            if (n < 0 || (size_t)n >= cap) continue;
            int id = man_bsearch(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++)
            tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    free(str_buffer);
    return 0;
}

static int encode_byte_bpe(bm_tokenizer_t* t, const char* text,
                           int* tokens, int* n_tokens) {
    const unsigned char* bytes = (const unsigned char*)text;
    size_t text_len = strlen(text);
    for (size_t i = 0; i < text_len; i++) {
        unsigned char b = bytes[i];
        int cp = t->b2u[b];
        char utf8[5];
        codepoint_to_utf8(cp, utf8);
        int id = man_bsearch(utf8, t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = (id != -1) ? id : b;
    }
    return bpe_merge(t, tokens, n_tokens);
}

static int encode_spm_bpe(bm_tokenizer_t* t, const char* text,
                          int* tokens, int* n_tokens) {
    char* norm = spm_normalize_bpe(text, t->prepend_space);
    if (!norm) return -1;

    const unsigned char* s = (const unsigned char*)norm;
    size_t len = strlen(norm);
    for (size_t i = 0; i < len;) {
        int adv = utf8_sequence_len(s + i);
        if ((size_t)adv > len - i) adv = 1;

        char buf[8];
        memcpy(buf, s + i, (size_t)adv);
        buf[adv] = '\0';
        int id = man_bsearch(buf, t->sorted_vocab, t->vocab_size);
        if (id >= 0) {
            tokens[(*n_tokens)++] = id;
        } else if (t->byte_fallback) {
            for (int j = 0; j < adv; j++) {
                char bf[8];
                snprintf(bf, sizeof(bf), "<0x%02X>", s[i + (size_t)j]);
                int bid = man_bsearch(bf, t->sorted_vocab, t->vocab_size);
                tokens[(*n_tokens)++] = (bid >= 0) ? bid : t->unk_id;
            }
        } else {
            tokens[(*n_tokens)++] = t->unk_id;
        }
        i += (size_t)adv;
    }

    free(norm);
    return bpe_merge(t, tokens, n_tokens);
}

static int encode_unigram(bm_tokenizer_t* t, const char* text,
                          int* tokens, int* n_tokens) {
    char* norm = spm_normalize_unigram(text, t->prepend_space, t->split_words);
    if (!norm) return -1;

    int n = (int)strlen(norm);
    float* best = malloc((size_t)(n + 1) * sizeof(float));
    int* prev = malloc((size_t)(n + 1) * sizeof(int));
    int* best_tok = malloc((size_t)(n + 1) * sizeof(int));
    int* rev = malloc((size_t)(n + 1) * sizeof(int));
    size_t maxlen = t->max_token_length;
    char* buf = malloc(maxlen + 1);
    if (!best || !prev || !best_tok || !rev || !buf) {
        free(best); free(prev); free(best_tok); free(rev); free(buf);
        free(norm);
        return -1;
    }

    for (int i = 0; i <= n; i++) {
        best[i] = -1e30f;
        prev[i] = -1;
        best_tok[i] = t->unk_id;
    }
    best[0] = 0.0f;

    for (int i = 0; i < n; i++) {
        if (best[i] < -1e29f) continue;

        int any = 0;
        int limit = (int)((size_t)(n - i) < maxlen ? (size_t)(n - i) : maxlen);
        for (int len = 1; len <= limit; len++) {
            memcpy(buf, norm + i, (size_t)len);
            buf[len] = '\0';
            int id = man_bsearch(buf, t->sorted_vocab, t->vocab_size);
            if (id >= 0 && t->vocab_scores[id] > -1e19f) {
                any = 1;
                float score = best[i] + t->vocab_scores[id];
                if (score > best[i + len]) {
                    best[i + len] = score;
                    prev[i + len] = i;
                    best_tok[i + len] = id;
                }
            }
        }

        if (t->byte_fallback) {
            char bf[8];
            snprintf(bf, sizeof(bf), "<0x%02X>", (unsigned char)norm[i]);
            int bid = man_bsearch(bf, t->sorted_vocab, t->vocab_size);
            if (bid >= 0) {
                any = 1;
                float score = best[i] + t->vocab_scores[bid];
                if (score > best[i + 1]) {
                    best[i + 1] = score;
                    prev[i + 1] = i;
                    best_tok[i + 1] = bid;
                }
            }
        }

        if (!any) {
            int unk_len = utf8_sequence_len((const unsigned char*)norm + i);
            if (i + unk_len > n) unk_len = 1;
            float score = best[i] - 100.0f;
            if (score > best[i + unk_len]) {
                best[i + unk_len] = score;
                prev[i + unk_len] = i;
                best_tok[i + unk_len] = t->unk_id;
            }
        }
    }

    if (best[n] < -1e29f) {
        free(best); free(prev); free(best_tok); free(rev); free(buf);
        free(norm);
        return -1;
    }

    int count = 0, pos = n;
    while (pos > 0 && prev[pos] >= 0) {
        rev[count++] = best_tok[pos];
        pos = prev[pos];
    }
    for (int i = count - 1; i >= 0; i--)
        tokens[(*n_tokens)++] = rev[i];

    free(best); free(prev); free(best_tok); free(rev); free(buf);
    free(norm);
    return 0;
}

int bm_tokenizer_encode(bm_tokenizer_t* t, const char* text, int8_t bos,
                        int8_t eos, int* tokens, int* n_tokens) {
    if (!text || !t->sorted_vocab) return -1;

    *n_tokens = 0;
    if (bos && t->bos_id >= 0) tokens[(*n_tokens)++] = t->bos_id;

    int rc;
    if (t->kind == BM_TOKENIZER_UNIGRAM)
        rc = encode_unigram(t, text, tokens, n_tokens);
    else if (t->kind == BM_TOKENIZER_SPM_BPE)
        rc = encode_spm_bpe(t, text, tokens, n_tokens);
    else
        rc = encode_byte_bpe(t, text, tokens, n_tokens);

    if (rc != 0) return rc;
    if (eos && t->eos_id >= 0) tokens[(*n_tokens)++] = t->eos_id;
    return 0;
}
