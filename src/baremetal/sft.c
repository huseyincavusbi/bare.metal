// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#define _POSIX_C_SOURCE 200809L

#include "baremetal/sft.h"
#include "baremetal/json.h"
#include "utils/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* s;
    size_t len;
    size_t cap;
} sb_t;

static int sb_reserve(sb_t* b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return 0;
    size_t cap = b->cap ? b->cap : 128;
    while (cap < b->len + extra + 1) cap *= 2;
    char* p = (char*)realloc(b->s, cap);
    if (!p) return -1;
    b->s = p;
    b->cap = cap;
    return 0;
}

static int sb_append(sb_t* b, const char* s) {
    size_t n = strlen(s);
    if (sb_reserve(b, n) != 0) return -1;
    memcpy(b->s + b->len, s, n + 1);
    b->len += n;
    return 0;
}

static char* sb_take(sb_t* b) {
    if (!b->s) {
        b->s = (char*)malloc(1);
        if (b->s) b->s[0] = '\0';
    }
    char* p = b->s;
    b->s = NULL;
    b->len = b->cap = 0;
    return p;
}

static void append_message(sb_t* b, bmt_sft_template_t tmpl, const char* role, const char* content) {
    if (tmpl == BMT_SFT_CHATML) {
        sb_append(b, "<|im_start|>");
        sb_append(b, role);
        sb_append(b, "\n");
        sb_append(b, content);
        sb_append(b, "<|im_end|>\n");
    } else {
        sb_append(b, "<|start_header_id|>");
        sb_append(b, role);
        sb_append(b, "<|end_header_id|>\n\n");
        sb_append(b, content);
        sb_append(b, "<|eot_id|>");
    }
}

static void append_assistant_header(sb_t* b, bmt_sft_template_t tmpl) {
    if (tmpl == BMT_SFT_CHATML) {
        sb_append(b, "<|im_start|>assistant\n");
    } else {
        sb_append(b, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
}

static void append_completion_end(sb_t* b, bmt_sft_template_t tmpl) {
    if (tmpl == BMT_SFT_CHATML) sb_append(b, "<|im_end|>\n");
    else sb_append(b, "<|eot_id|>");
}

int bmt_sft_parse_template(const char* name) {
    if (!name) return -1;
    if (strcmp(name, "chatml") == 0) return BMT_SFT_CHATML;
    if (strcmp(name, "llama3") == 0) return BMT_SFT_LLAMA3;
    return -1;
}

int bmt_sft_detect_template(const char* model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) return BMT_SFT_CHATML;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return BMT_SFT_CHATML; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return BMT_SFT_CHATML;
    }
    buf[len] = '\0';
    fclose(f);
    int tmpl = BMT_SFT_CHATML;
    if (strstr(buf, "<|start_header_id|>")) tmpl = BMT_SFT_LLAMA3;
    free(buf);
    return tmpl;
}

static int render_messages(bmt_sft_template_t tmpl, const char** roles,
                           const char** contents, int n, int last_assistant,
                           bmt_sft_pair_t* pair) {
    (void)n;
    sb_t prompt = {0}, completion = {0};
    if (tmpl == BMT_SFT_LLAMA3) sb_append(&prompt, "<|begin_of_text|>");
    for (int i = 0; i < last_assistant; i++)
        append_message(&prompt, tmpl, roles[i], contents[i]);
    append_assistant_header(&prompt, tmpl);

    sb_append(&completion, contents[last_assistant]);
    append_completion_end(&completion, tmpl);

    pair->prompt = sb_take(&prompt);
    pair->completion = sb_take(&completion);
    return (pair->prompt && pair->completion) ? 0 : -1;
}

static int render_prompt_completion(bmt_sft_template_t tmpl,
                                    const char* prompt, const char* completion,
                                    bmt_sft_pair_t* pair) {
    sb_t p = {0}, c = {0};
    if (tmpl == BMT_SFT_LLAMA3) sb_append(&p, "<|begin_of_text|>");
    append_message(&p, tmpl, "user", prompt);
    append_assistant_header(&p, tmpl);
    sb_append(&c, completion);
    append_completion_end(&c, tmpl);
    pair->prompt = sb_take(&p);
    pair->completion = sb_take(&c);
    return (pair->prompt && pair->completion) ? 0 : -1;
}

static int parse_example(const char* line, bmt_sft_template_t tmpl, bmt_sft_pair_t* pair) {
    char err[128];
    bmt_json_t* root = bmt_json_parse(line, err, sizeof(err));
    if (!root) return -1;

    const bmt_json_t* msgs = bmt_json_get(root, "messages");
    int ok = -1;
    if (msgs && bmt_json_type(msgs) == BMT_JSON_ARRAY) {
        int n = bmt_json_array_count(msgs);
        const char** roles = (const char**)calloc((size_t)n, sizeof(char*));
        const char** contents = (const char**)calloc((size_t)n, sizeof(char*));
        int last_assistant = -1;
        if (roles && contents) {
            for (int i = 0; i < n; i++) {
                const bmt_json_t* m = bmt_json_array_get(msgs, i);
                if (!m || bmt_json_type(m) != BMT_JSON_OBJECT) continue;
                roles[i] = bmt_json_string(bmt_json_get(m, "role"));
                contents[i] = bmt_json_string(bmt_json_get(m, "content"));
                if (!roles[i] || !contents[i]) continue;
                if (strcmp(roles[i], "assistant") == 0) last_assistant = i;
            }
            if (last_assistant >= 0)
                ok = render_messages(tmpl, roles, contents, n, last_assistant, pair);
        }
        free(roles);
        free(contents);
    } else {
        const char* prompt = bmt_json_string(bmt_json_get(root, "prompt"));
        const char* completion = bmt_json_string(bmt_json_get(root, "completion"));
        if (prompt && completion && prompt[0] && completion[0])
            ok = render_prompt_completion(tmpl, prompt, completion, pair);
    }
    bmt_json_free(root);
    return ok;
}

int bmt_sft_load_jsonl(const char* path, bmt_sft_template_t tmpl, int max_examples,
                       bmt_sft_dataset_t* out, int* n_skipped) {
    if (!path || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (n_skipped) *n_skipped = 0;

    FILE* f = fopen(path, "rb");
    if (!f) {
        BMT_LOG_ERROR("bmt_sft_load_jsonl: cannot open %s", path);
        return -1;
    }
    int cap = 16;
    out->items = (bmt_sft_pair_t*)calloc((size_t)cap, sizeof(bmt_sft_pair_t));
    if (!out->items) { fclose(f); return -1; }

    char* line = NULL;
    size_t line_cap = 0;
    ssize_t len;
    while ((len = getline(&line, &line_cap, f)) >= 0) {
        (void)len;
        bmt_sft_pair_t pair = {0};
        if (parse_example(line, tmpl, &pair) != 0) {
            if (n_skipped) (*n_skipped)++;
            continue;
        }
        if (out->n == cap) {
            cap *= 2;
            bmt_sft_pair_t* p = (bmt_sft_pair_t*)realloc(out->items, (size_t)cap * sizeof(bmt_sft_pair_t));
            if (!p) { free(pair.prompt); free(pair.completion); break; }
            out->items = p;
        }
        out->items[out->n++] = pair;
        if (max_examples > 0 && out->n >= max_examples) break;
    }
    free(line);
    fclose(f);
    return out->n > 0 ? 0 : -1;
}

void bmt_sft_free_dataset(bmt_sft_dataset_t* d) {
    if (!d) return;
    for (int i = 0; i < d->n; i++) {
        free(d->items[i].prompt);
        free(d->items[i].completion);
    }
    free(d->items);
    memset(d, 0, sizeof(*d));
}

static int encode_text(bm_tokenizer_t* tok, const char* text, int** out, int* n_out) {
    size_t cap = strlen(text) * 4 + 256;
    int* ids = (int*)malloc(cap * sizeof(int));
    if (!ids) return -1;
    int n = 0;
    if (bm_tokenizer_encode(tok, text, 0, 0, ids, &n) != 0) {
        free(ids);
        return -1;
    }
    *out = ids;
    *n_out = n;
    return 0;
}

int bmt_sft_build_example(bm_tokenizer_t* tok, const bmt_sft_pair_t* pair, int S,
                          int* inputs, int* targets, uint8_t* attn, uint8_t* loss) {
    if (!tok || !pair || !inputs || !targets || !attn || !loss || S < 2) return -1;
    int *ptok = NULL, *ctok = NULL;
    int pn = 0, cn = 0;
    if (encode_text(tok, pair->prompt, &ptok, &pn) != 0) return -1;
    if (encode_text(tok, pair->completion, &ctok, &cn) != 0) { free(ptok); return -1; }
    if (cn <= 0) { free(ptok); free(ctok); return -1; }

    int cc = cn > S ? S : cn;
    int pk = pn;
    if (pk + cc > S) pk = S - cc;
    if (pk < 0) pk = 0;
    int real = pk + cc;
    if (real <= 0) { free(ptok); free(ctok); return -1; }

    int* seq = (int*)malloc((size_t)(S + 1) * sizeof(int));
    if (!seq) { free(ptok); free(ctok); return -1; }
    for (int i = 0; i < pk; i++) seq[i] = ptok[pn - pk + i];
    for (int i = 0; i < cc; i++) seq[pk + i] = ctok[i];
    int eos = tok->eos_id >= 0 ? tok->eos_id : 0;
    for (int i = real; i <= S; i++) seq[i] = eos;

    for (int i = 0; i < S; i++) {
        inputs[i] = seq[i];
        targets[i] = seq[i + 1];
        attn[i] = (uint8_t)(i < real ? 1 : 0);
        loss[i] = (uint8_t)((i + 1 >= pk && i + 1 < real) ? 1 : 0);
    }
    free(seq);
    free(ptok);
    free(ctok);
    return 0;
}
