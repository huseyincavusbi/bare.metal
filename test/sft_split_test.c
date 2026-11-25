#include "baremetal/sft.h"
#include <stdio.h>

static int failures;

static void check(int ok, const char* what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static int contains(const int* a, int n, int v) {
    for (int i = 0; i < n; i++) if (a[i] == v) return 1;
    return 0;
}

int main(void) {
    int train[32], val[32];
    int nt = 0, nv = 0, nh = 0;

    check(bmt_sft_split_indices(20, 10, 0, train, &nt, val, &nv, &nh) == 0,
          "split without limit");
    check(nt == 18 && nv == 2 && nh == 2, "all held-out examples evaluated");
    check(val[0] == 0 && val[1] == 10, "held-out indices are deterministic");
    check(!contains(train, nt, 0) && !contains(train, nt, 10),
          "held-out indices excluded from training");

    check(bmt_sft_split_indices(20, 10, 1, train, &nt, val, &nv, &nh) == 0,
          "split with limit");
    check(nt == 18 && nv == 1 && nh == 2, "limit caps evaluation only");
    check(val[0] == 0, "limit keeps the first held-out index");
    check(!contains(train, nt, 0) && !contains(train, nt, 10),
          "unseen held-out examples still excluded from training");

    check(bmt_sft_split_indices(20, 10, 100, train, &nt, val, &nv, &nh) == 0,
          "limit above holdout");
    check(nt == 18 && nv == 2 && nh == 2, "limit above holdout is harmless");

    check(bmt_sft_split_indices(20, 0, 0, train, &nt, val, &nv, &nh) == 0,
          "split disabled");
    check(nt == 20 && nv == 0 && nh == 0, "all examples train without a split");

    check(bmt_sft_split_indices(0, 10, 0, train, &nt, val, &nv, &nh) == 0,
          "empty dataset");
    check(nt == 0 && nv == 0 && nh == 0, "empty dataset has no indices");

    check(bmt_sft_split_indices(20, 10, 0, NULL, &nt, val, &nv, &nh) == -1,
          "missing training array rejected");
    check(bmt_sft_split_indices(20, 10, 0, train, NULL, val, &nv, &nh) == -1,
          "missing count rejected");

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all SFT split tests passed\n");
    return 0;
}
