/* json_test.c — minimal JSON parser used for config.json loading. */
#include "baremetal/json.h"
#include <stdio.h>
#include <string.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static void expect_parse_fail(const char* name, const char* text) {
    char err[160];
    bmt_json_t* v = bmt_json_parse(text, err, sizeof(err));
    printf("%s: %s rejected (%s)\n", v ? "FAIL" : "PASS", name,
           err[0] ? err : "no error");
    if (v) {
        bmt_json_free(v);
        failures++;
    }
}

int main(void) {
    const char* doc =
        "{"
        "  \"name\": \"smollm\","
        "  \"n\": 42,"
        "  \"neg\": -1.5e3,"
        "  \"ok\": true,"
        "  \"no\": false,"
        "  \"nothing\": null,"
        "  \"list\": [1, 2, 3],"
        "  \"nested\": {\"a\": {\"b\": \"c\"}},"
        "  \"esc\": \"a\\\"b\\\\c\\n\\u00e9\","
        "  \"emoji\": \"\\uD83D\\uDE00\","
        "  \"empty\": {}"
        "}";

    char err[160];
    bmt_json_t* root = bmt_json_parse(doc, err, sizeof(err));
    check(root != NULL, "parse valid config object");
    if (!root) {
        printf("error: %s\n", err);
        return 1;
    }

    check(strcmp(bmt_json_string(bmt_json_get(root, "name")), "smollm") == 0,
          "string value");
    int n = 0;
    check(bmt_json_int(bmt_json_get(root, "n"), &n) == 0 && n == 42,
          "integer value");
    float f = 0.0f;
    check(bmt_json_float(bmt_json_get(root, "neg"), &f) == 0 && f == -1500.0f,
          "exponent value");
    int b = -1;
    check(bmt_json_bool(bmt_json_get(root, "ok"), &b) == 0 && b == 1,
          "true value");
    check(bmt_json_bool(bmt_json_get(root, "no"), &b) == 0 && b == 0,
          "false value");
    check(bmt_json_type(bmt_json_get(root, "nothing")) == BMT_JSON_NULL,
          "null value");

    const bmt_json_t* list = bmt_json_get(root, "list");
    check(bmt_json_array_count(list) == 3, "array count");
    int item = 0;
    check(bmt_json_int(bmt_json_array_get(list, 1), &item) == 0 && item == 2,
          "array element");

    const bmt_json_t* nested = bmt_json_get(root, "nested");
    const bmt_json_t* a = bmt_json_get(nested, "a");
    check(strcmp(bmt_json_string(bmt_json_get(a, "b")), "c") == 0,
          "nested object");

    check(strcmp(bmt_json_string(bmt_json_get(root, "esc")),
                 "a\"b\\c\n\xC3\xA9") == 0, "escapes");
    check(strcmp(bmt_json_string(bmt_json_get(root, "emoji")),
                 "\xF0\x9F\x98\x80") == 0, "surrogate pair");
    check(bmt_json_get(root, "missing") == NULL, "missing key");
    check(bmt_json_array_count(bmt_json_get(root, "missing")) == 0,
          "missing key is not an array");

    bmt_json_free(root);

    expect_parse_fail("empty input", "");
    expect_parse_fail("unterminated object", "{\"a\": 1");
    expect_parse_fail("trailing characters", "{\"a\": 1} x");
    expect_parse_fail("invalid escape", "{\"a\": \"\\q\"}");
    expect_parse_fail("leading zero", "{\"a\": 01}");
    expect_parse_fail("trailing comma", "{\"a\": [1,]}");
    expect_parse_fail("bad literal", "{\"a\": tru}");
    expect_parse_fail("bad number", "{\"a\": -}");

    if (failures) {
        printf("%d FAILURE(S)\n", failures);
        return 1;
    }
    printf("all json tests passed\n");
    return 0;
}
