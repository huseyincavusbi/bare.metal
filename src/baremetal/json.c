// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "baremetal/json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BMT_JSON_MAX_DEPTH 64

struct bmt_json_s {
    bmt_json_type_t type;
    union {
        int boolean;
        double number;
        char* string;
        struct {
            bmt_json_t** items;
            int count;
        } array;
        struct {
            char** keys;
            bmt_json_t** values;
            int count;
        } object;
    } as;
};

typedef struct {
    const char* start;
    const char* p;
    const char* end;
    char* err;
    size_t err_len;
    int depth;
} bmt_json_parser_t;

static void json_error(bmt_json_parser_t* p, const char* msg) {
    if (p->err && p->err_len)
        snprintf(p->err, p->err_len, "%s at offset %ld", msg, (long)(p->p - p->start));
}

static void skip_ws(bmt_json_parser_t* p) {
    while (p->p < p->end && (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r'))
        p->p++;
}

static bmt_json_t* json_new(bmt_json_parser_t* p, bmt_json_type_t type) {
    bmt_json_t* v = calloc(1, sizeof(*v));
    if (!v) json_error(p, "out of memory");
    else v->type = type;
    return v;
}

static int hex4(const char* s) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

static int utf8_encode(unsigned cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static char* parse_string_raw(bmt_json_parser_t* p) {
    p->p++;
    const char* s = p->p;

    while (p->p < p->end) {
        unsigned char c = (unsigned char)*p->p;
        if (c == '\\') {
            p->p++;
            if (p->p >= p->end) {
                json_error(p, "unterminated escape");
                return NULL;
            }
            p->p++;
            continue;
        }
        if (c == '"') break;
        if (c < 0x20) {
            json_error(p, "control character in string");
            return NULL;
        }
        p->p++;
    }

    if (p->p >= p->end) {
        json_error(p, "unterminated string");
        return NULL;
    }

    size_t raw_len = (size_t)(p->p - s);
    char* out = malloc(raw_len + 1);
    if (!out) {
        json_error(p, "out of memory");
        return NULL;
    }

    const char* end = s + raw_len;
    const char* q = s;
    char* d = out;
    while (q < end) {
        if (*q != '\\') {
            *d++ = *q++;
            continue;
        }
        q++;
        if (q >= end) goto bad_escape;
        switch (*q) {
            case '"':  *d++ = '"';  q++; break;
            case '\\': *d++ = '\\'; q++; break;
            case '/':  *d++ = '/';  q++; break;
            case 'b':  *d++ = '\b'; q++; break;
            case 'f':  *d++ = '\f'; q++; break;
            case 'n':  *d++ = '\n'; q++; break;
            case 'r':  *d++ = '\r'; q++; break;
            case 't':  *d++ = '\t'; q++; break;
            case 'u': {
                if (q + 4 >= end) goto bad_escape;
                int cp = hex4(q + 1);
                if (cp < 0) goto bad_escape;
                q += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (end - q >= 6 && q[0] == '\\' && q[1] == 'u') {
                        int lo = hex4(q + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            q += 6;
                        } else {
                            cp = 0xFFFD;
                        }
                    } else {
                        cp = 0xFFFD;
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    cp = 0xFFFD;
                }
                d += utf8_encode((unsigned)cp, d);
                break;
            }
            default:
                goto bad_escape;
        }
    }
    *d = '\0';
    p->p++;
    return out;

bad_escape:
    json_error(p, "invalid escape in string");
    free(out);
    return NULL;
}

static bmt_json_t* parse_value(bmt_json_parser_t* p);

static bmt_json_t* parse_array(bmt_json_parser_t* p) {
    if (++p->depth > BMT_JSON_MAX_DEPTH) {
        json_error(p, "nesting too deep");
        p->depth--;
        return NULL;
    }

    bmt_json_t* v = json_new(p, BMT_JSON_ARRAY);
    if (!v) { p->depth--; return NULL; }

    p->p++;
    skip_ws(p);
    if (p->p < p->end && *p->p == ']') {
        p->p++;
        p->depth--;
        return v;
    }

    while (1) {
        bmt_json_t* item = parse_value(p);
        if (!item) { bmt_json_free(v); p->depth--; return NULL; }

        bmt_json_t** items = realloc(v->as.array.items,
                                     (size_t)(v->as.array.count + 1) * sizeof(*items));
        if (!items) {
            bmt_json_free(item);
            bmt_json_free(v);
            json_error(p, "out of memory");
            p->depth--;
            return NULL;
        }
        v->as.array.items = items;
        v->as.array.items[v->as.array.count++] = item;

        skip_ws(p);
        if (p->p < p->end && *p->p == ',') { p->p++; continue; }
        if (p->p < p->end && *p->p == ']') { p->p++; break; }
        json_error(p, "expected ',' or ']'");
        bmt_json_free(v);
        p->depth--;
        return NULL;
    }

    p->depth--;
    return v;
}

static bmt_json_t* parse_object(bmt_json_parser_t* p) {
    if (++p->depth > BMT_JSON_MAX_DEPTH) {
        json_error(p, "nesting too deep");
        p->depth--;
        return NULL;
    }

    bmt_json_t* v = json_new(p, BMT_JSON_OBJECT);
    if (!v) { p->depth--; return NULL; }

    p->p++;
    skip_ws(p);
    if (p->p < p->end && *p->p == '}') {
        p->p++;
        p->depth--;
        return v;
    }

    while (1) {
        skip_ws(p);
        if (p->p >= p->end || *p->p != '"') {
            json_error(p, "expected object key");
            bmt_json_free(v);
            p->depth--;
            return NULL;
        }
        char* key = parse_string_raw(p);
        if (!key) { bmt_json_free(v); p->depth--; return NULL; }

        skip_ws(p);
        if (p->p >= p->end || *p->p != ':') {
            json_error(p, "expected ':'");
            free(key);
            bmt_json_free(v);
            p->depth--;
            return NULL;
        }
        p->p++;

        bmt_json_t* val = parse_value(p);
        if (!val) { free(key); bmt_json_free(v); p->depth--; return NULL; }

        int n = v->as.object.count;
        char** keys = realloc(v->as.object.keys, (size_t)(n + 1) * sizeof(*keys));
        bmt_json_t** values = realloc(v->as.object.values, (size_t)(n + 1) * sizeof(*values));
        if (!keys || !values) {
            free(keys ? keys : v->as.object.keys);
            free(values ? values : v->as.object.values);
            v->as.object.keys = NULL;
            v->as.object.values = NULL;
            v->as.object.count = 0;
            free(key);
            bmt_json_free(val);
            bmt_json_free(v);
            json_error(p, "out of memory");
            p->depth--;
            return NULL;
        }
        v->as.object.keys = keys;
        v->as.object.values = values;
        v->as.object.keys[n] = key;
        v->as.object.values[n] = val;
        v->as.object.count = n + 1;

        skip_ws(p);
        if (p->p < p->end && *p->p == ',') { p->p++; continue; }
        if (p->p < p->end && *p->p == '}') { p->p++; break; }
        json_error(p, "expected ',' or '}'");
        bmt_json_free(v);
        p->depth--;
        return NULL;
    }

    p->depth--;
    return v;
}

static int scan_number(bmt_json_parser_t* p, size_t* len) {
    const char* q = p->p;
    const char* end = p->end;

    if (q < end && *q == '-') q++;
    if (q >= end) return -1;

    if (*q == '0') {
        q++;
    } else if (*q >= '1' && *q <= '9') {
        while (q < end && *q >= '0' && *q <= '9') q++;
    } else {
        return -1;
    }

    if (q < end && *q == '.') {
        q++;
        if (q >= end || *q < '0' || *q > '9') return -1;
        while (q < end && *q >= '0' && *q <= '9') q++;
    }

    if (q < end && (*q == 'e' || *q == 'E')) {
        q++;
        if (q < end && (*q == '+' || *q == '-')) q++;
        if (q >= end || *q < '0' || *q > '9') return -1;
        while (q < end && *q >= '0' && *q <= '9') q++;
    }

    *len = (size_t)(q - p->p);
    return 0;
}

static bmt_json_t* parse_number(bmt_json_parser_t* p) {
    size_t len = 0;
    if (scan_number(p, &len) != 0) {
        json_error(p, "invalid number");
        return NULL;
    }

    char buf[64];
    if (len >= sizeof(buf)) {
        json_error(p, "number too long");
        return NULL;
    }
    memcpy(buf, p->p, len);
    buf[len] = '\0';

    bmt_json_t* v = json_new(p, BMT_JSON_NUMBER);
    if (!v) return NULL;
    v->as.number = strtod(buf, NULL);
    p->p += len;
    return v;
}

static bmt_json_t* parse_value(bmt_json_parser_t* p) {
    skip_ws(p);
    if (p->p >= p->end) {
        json_error(p, "unexpected end of input");
        return NULL;
    }

    switch (*p->p) {
        case '{': return parse_object(p);
        case '[': return parse_array(p);
        case '"': {
            char* s = parse_string_raw(p);
            if (!s) return NULL;
            bmt_json_t* v = json_new(p, BMT_JSON_STRING);
            if (!v) { free(s); return NULL; }
            v->as.string = s;
            return v;
        }
        case 't':
            if (p->end - p->p >= 4 && strncmp(p->p, "true", 4) == 0) {
                bmt_json_t* v = json_new(p, BMT_JSON_BOOL);
                if (!v) return NULL;
                v->as.boolean = 1;
                p->p += 4;
                return v;
            }
            break;
        case 'f':
            if (p->end - p->p >= 5 && strncmp(p->p, "false", 5) == 0) {
                bmt_json_t* v = json_new(p, BMT_JSON_BOOL);
                if (!v) return NULL;
                v->as.boolean = 0;
                p->p += 5;
                return v;
            }
            break;
        case 'n':
            if (p->end - p->p >= 4 && strncmp(p->p, "null", 4) == 0) {
                bmt_json_t* v = json_new(p, BMT_JSON_NULL);
                if (!v) return NULL;
                p->p += 4;
                return v;
            }
            break;
        default:
            if (*p->p == '-' || (*p->p >= '0' && *p->p <= '9'))
                return parse_number(p);
            break;
    }

    json_error(p, "unexpected token");
    return NULL;
}

bmt_json_t* bmt_json_parse(const char* text, char* err, size_t err_len) {
    bmt_json_parser_t p;
    p.start = text;
    p.p = text;
    p.end = text + strlen(text);
    p.err = err;
    p.err_len = err_len;
    p.depth = 0;
    if (err && err_len) err[0] = '\0';

    skip_ws(&p);
    if (p.p >= p.end) {
        json_error(&p, "empty input");
        return NULL;
    }

    bmt_json_t* v = parse_value(&p);
    if (!v) return NULL;

    skip_ws(&p);
    if (p.p != p.end) {
        json_error(&p, "trailing characters");
        bmt_json_free(v);
        return NULL;
    }
    return v;
}

void bmt_json_free(bmt_json_t* value) {
    if (!value) return;
    switch (value->type) {
        case BMT_JSON_STRING:
            free(value->as.string);
            break;
        case BMT_JSON_ARRAY:
            for (int i = 0; i < value->as.array.count; i++)
                bmt_json_free(value->as.array.items[i]);
            free(value->as.array.items);
            break;
        case BMT_JSON_OBJECT:
            for (int i = 0; i < value->as.object.count; i++) {
                free(value->as.object.keys[i]);
                bmt_json_free(value->as.object.values[i]);
            }
            free(value->as.object.keys);
            free(value->as.object.values);
            break;
        default:
            break;
    }
    free(value);
}

bmt_json_type_t bmt_json_type(const bmt_json_t* value) {
    return value ? value->type : BMT_JSON_NULL;
}

const bmt_json_t* bmt_json_get(const bmt_json_t* object, const char* key) {
    if (!object || object->type != BMT_JSON_OBJECT) return NULL;
    for (int i = 0; i < object->as.object.count; i++) {
        if (strcmp(object->as.object.keys[i], key) == 0)
            return object->as.object.values[i];
    }
    return NULL;
}

int bmt_json_int(const bmt_json_t* value, int* out) {
    if (!value || value->type != BMT_JSON_NUMBER) return -1;
    *out = (int)value->as.number;
    return 0;
}

int bmt_json_float(const bmt_json_t* value, float* out) {
    if (!value || value->type != BMT_JSON_NUMBER) return -1;
    *out = (float)value->as.number;
    return 0;
}

int bmt_json_bool(const bmt_json_t* value, int* out) {
    if (!value || value->type != BMT_JSON_BOOL) return -1;
    *out = value->as.boolean;
    return 0;
}

const char* bmt_json_string(const bmt_json_t* value) {
    if (!value || value->type != BMT_JSON_STRING) return NULL;
    return value->as.string;
}

int bmt_json_array_count(const bmt_json_t* value) {
    if (!value || value->type != BMT_JSON_ARRAY) return 0;
    return value->as.array.count;
}

const bmt_json_t* bmt_json_array_get(const bmt_json_t* value, int index) {
    if (!value || value->type != BMT_JSON_ARRAY) return NULL;
    if (index < 0 || index >= value->as.array.count) return NULL;
    return value->as.array.items[index];
}

int bmt_json_object_count(const bmt_json_t* value) {
    if (!value || value->type != BMT_JSON_OBJECT) return 0;
    return value->as.object.count;
}

const char* bmt_json_object_key(const bmt_json_t* value, int index) {
    if (!value || value->type != BMT_JSON_OBJECT) return NULL;
    if (index < 0 || index >= value->as.object.count) return NULL;
    return value->as.object.keys[index];
}

const bmt_json_t* bmt_json_object_value(const bmt_json_t* value, int index) {
    if (!value || value->type != BMT_JSON_OBJECT) return NULL;
    if (index < 0 || index >= value->as.object.count) return NULL;
    return value->as.object.values[index];
}
