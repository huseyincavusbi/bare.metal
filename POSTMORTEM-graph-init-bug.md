# Postmortem: Uninitialized `quantized` Field Causing Nondeterministic Training

**Project:** `bare.metal` — a from-scratch Metal (MSL + C11) engine for SmolLM2-135M inference and training on Apple Silicon.
**Symptom:** Training was nondeterministic — step-0 loss was bimodal (`5.0104` vs `4.9815`), runs occasionally diverged to `NaN`, and later outright crashed with `SIGBUS`.
**Root cause:** `bmt_graph_add_tensor()` never initialized the `quantized` field. For any tensor created after the graph's tensor array grew past its initial capacity, `quantized` held **heap garbage**, causing 33 weight buffers to be allocated as **Q8** but written as **bf16** — an out-of-bounds write.
**Fix:** commit `f44c0aa` — zero-initialize tensor/node structs and the `realloc`'d tail in `graph.c`.
**Status:** Fixed, validated, CI green.

---

## 1. Summary

For several days the training gate (`g4_test`) was flaky: with byte-identical inputs, weights, and code, the first-step loss would come up either `5.0104` (healthy) or `4.9815` (which frequently diverged to `NaN`), and the rate swung from ~2% to ~100% over time. The bug survived a machine reboot, ruled out environment and toolchain. It was eventually traced to a single uninitialized struct field that caused a buffer size mismatch and out-of-bounds writes.

The fix is three lines of defensive initialization.

---

## 2. Impact

- Training results were nondeterministic and occasionally produced `NaN`, corrupting weights.
- The CI gate was unreliable (roughly 50% failure), blocking PR #2.
- The bug also manifested as intermittent `SIGBUS` crashes (signal 10) at `-O2`.
- Inference (`g3`) happened to look deterministic in most runs, masking the underlying defect.

---

## 3. Root cause

### 3.1 The struct

`bmt_tensor_t` (`src/baremetal/graph.h`) ends with:

```c
typedef struct {
    int id;
    bmt_tensor_type_t type;
    int dims[BMT_MAX_TENSOR_DIMS];
    int n_dims;
    void* weight_ptr;
    int   quantized;   /* 1 = upload as Q8 blocks */
} bmt_tensor_t;
```

### 3.2 Graph storage growth

`bmt_graph_create()` (`src/baremetal/graph.c`) starts with capacity 128 and `calloc`s the arrays:

```c
g->capacity_tensors = 128;
g->tensors = calloc(g->capacity_tensors, sizeof(bmt_tensor_t));   // zeroed
```

`bmt_graph_add_tensor()` grew the array with **`realloc`**, which does **not** zero the newly added region, and then set every field **except `quantized`**:

```c
int bmt_graph_add_tensor(bmt_graph_t* graph, bmt_tensor_type_t type, int n_dims, const int* dims) {
    if (graph->n_tensors >= graph->capacity_tensors) {
        graph->capacity_tensors *= 2;
        graph->tensors = realloc(graph->tensors, graph->capacity_tensors * sizeof(bmt_tensor_t));
    }
    int id = graph->n_tensors++;
    bmt_tensor_t* t = &graph->tensors[id];
    t->id = id;
    t->type = type;
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims && i < BMT_MAX_TENSOR_DIMS; i++) t->dims[i] = dims[i];
    t->weight_ptr = NULL;
    // t->quantized  <-- NEVER SET
    return id;
}
```

So:
- Tensors `0..127` live in the initial `calloc`'d region → `quantized == 0` ✓
- Tensors `>= 128` live in `realloc`'d memory → `quantized == <heap garbage>` ✗

SmolLM2-135M's training graph has ~660 tensors (272 of them weights), so **most** weights were affected.

### 3.3 Consequence: buffer size mismatch → OOB writes

In `bmt_scheduler_create_with_precision()` (`src/baremetal/scheduler.c`), the upload path is chosen per tensor:

```c
int do_q8 = (t->type == BMT_TENSOR_TYPE_WEIGHT && t->weight_ptr && t->quantized
             && t->n_dims == 2 && (t->dims[1] % 32 == 0));

int do_bf16 = (!do_q8 && precision == BM_PRECISION_BF16
               && t->type == BMT_TENSOR_TYPE_WEIGHT && t->weight_ptr
               && t->n_dims >= 2);

if (do_q8) {
    size_t qbytes = bmt_q8_bytes(n_elems);          // (n/32) * sizeof(q8_block_t) = n * 1.125
    sched->buffers[i] = backend_buffer_alloc(backend, qbytes);
    bmt_quantize_q8(...);
} else if (do_bf16) {
    size_t bytes = n_elems * sizeof(bm_bf16_t);     // n * 2
    sched->buffers[i] = backend_buffer_alloc(backend, bytes);
    bm_f32_to_bf16_array(...);
}
```

Because `quantized` was garbage (non-zero), `do_q8` was **true** for 33 weight tensors even though the run was bf16. Their buffers were allocated Q8-sized (`n × 1.125` bytes) but then:

- **At create:** `bmt_quantize_q8()` and later the per-step `bm_f32_to_bf16_array()` wrote `n × 2` bytes into a `n × 1.125` buffer → OOB.
- **At every optimizer step:** `adamw_apply()` re-derives the working buffer from the fp32 master via `bm_f32_to_bf16_array(master, work, n)`, again writing `n × 2` into the undersized buffer.

Concrete observed values for tensor 130 (`n = 884736 = 1536 × 576`):

```
alloc = 995328   = (884736 / 32) * 36      // Q8 size
need  = 1769472  = 884736 * 2              // bf16 size
```

The ratio `995328 / 884736 = 1.125 = 36/32` is the Q8 block size — the fingerprint that identified the bug.

### 3.4 Why the symptom varied

The garbage value of `quantized` depended on prior heap contents, which vary between processes and over time. Depending on whether the garbage happened to be zero, the affected tensors took the Q8 path (broken) or the correct path — hence bimodality, temporal swings, and irreproducibility.

---

## 4. The fix

`src/baremetal/graph.c` — zero the newly grown region and each new struct:

```c
int bmt_graph_add_tensor(bmt_graph_t* graph, bmt_tensor_type_t type, int n_dims, const int* dims) {
    if (graph->n_tensors >= graph->capacity_tensors) {
        int old_cap = graph->capacity_tensors;
        graph->capacity_tensors *= 2;
        graph->tensors = realloc(graph->tensors, graph->capacity_tensors * sizeof(bmt_tensor_t));
        memset(&graph->tensors[old_cap], 0, (graph->capacity_tensors - old_cap) * sizeof(bmt_tensor_t));
    }
    int id = graph->n_tensors++;
    bmt_tensor_t* t = &graph->tensors[id];
    memset(t, 0, sizeof(*t));
    t->id = id;
    ...
}
```

The same treatment was applied to `bmt_graph_add_node()` (same `realloc` pattern, and it left trailing `inputs`/`params`/`fparams` slots uninitialized).

---

## 5. Investigation timeline (what we ruled out, and how)

The bug's intermittency sent us down many false paths. In roughly chronological order:

1. **Tokenizer / input data differs?** No — first-8 token IDs and tokenizer file hashes matched between machines and runs.
2. **Model weights differ?** No — `model.safetensors` sha256 matched HF; host fp32 weights hashed identically across runs.
3. **bf16 precision path bug?** The engine was not even using bf16 — `model->precision` was never set in `g4_test`, so it defaulted to FP32. (Later making bf16 the default *appeared* to fix it — actually it changed which tensors took the broken Q8 path.)
4. **Uninitialized activation buffers?** Zeroing all forward buffers did not fix the bimodality.
5. **Kernel races?** Every training-path kernel was audited — all are single-thread-per-output with serial loops and no shared-memory races (`matmul_forward_naive`, `rmsnorm_forward`, `swiglu_forward`, `rope_forward_seq`, `attention_forward_seq`).
6. **Environment / GPU state?** A machine **reboot** did not help → ruled out thermal/background-load/driver-state theories.
7. **Trainer teardown / stale Metal memory?** In-process retries were worse, but comprehensive zeroing and clean teardown did not fix it.
8. **Weight-buffer divergence.** Custom FNV byte-hash instrumentation (`BM_DEBUG_FORWARD`) showed that, in a bad run, weight tensor **130**'s GPU bytes differed from the (identical) host source — and that the first divergent activation was the matmul reading tensor 130. This was real, but an early attempt to interpret it was misled by hashing bf16-sized ranges on fp32 buffers (a byte-count artifact).
9. **AddressSanitizer?** Clean — because Metal shared buffers are not malloc-backed and are invisible to ASan.
10. **SIGBUS crash** appeared intermittently at `-O2` (and never at `-O0`/`-O1`). macOS crash reports and `lldb` localized it to a vectorized **store** in `bm_train_step` crossing a page boundary.
11. **Targeted instrumentation.** Adding per-phase markers (`BM_MARK`) localized it to `adamw_apply()`. A size assertion (`BM_CHECK_ADAMW`) fired:

    ```
    ADAMW OOB wid=130 n=884736 alloc=995328 need=1769472
    ```

    The consistent `1.125` ratio pointed at Q8.
12. **Flag inspection** (`BM_CHECK_SCHED`) showed the smoking gun:

    ```
    SCHED i=130 quant=572533860 q8=1 bf16=0 n=884736 alloc=995328
    ```

    `quant = 572533860` is heap garbage. Tracing where `quantized` is set led directly to `bmt_graph_add_tensor` and the `realloc` boundary at 128 — matching the "first divergence at tensor 130" observation from days earlier.

---

## 6. Validation

After the fix (`f44c0aa`):

- **FP32 determinism restored:** 5/5 runs identical (`5.0104 → 4.6078`), no bimodality, no `NaN`.
- **g4 gate:** 8/8 PASS, 0 crashes (previously ~50% failure).
- **Buffer sizing correct:**
  ```
  SCHED i=130 quant=0 q8=0 bf16=1 n=884736 alloc=1769472
  ```
- **Inference sanity:** coherent SmolLM2 output.
- **CI (GitHub Actions, `macos-latest`, PR #2): fully green** — run `35346816133`:
  - Build (inference + training) ✓
  - Golden refs (PyTorch) ✓
  - Kernel tests ✓
  - g3 (loss curve + weights vs PyTorch) ✓
  - g4 (200-step loss decrease) ✓
  - Inference E2E: fp32, bf16, fp16, q8 ✓

Supporting commits on the `ci` branch:

| Commit | Description |
|---|---|
| `f44c0aa` | **fix:** zero-initialize graph tensors/nodes on realloc |
| `f18073a` | test: g4 uses production config and median-based check |
| `22c2cfc` | feat: default train precision to bf16 (device-supported) |
| `20d6975` | test: g4 uses device-supported precision |
| `3331699` | test: add reference gradient dump generator |

---

## 7. Lessons learned / preventive measures

1. **`realloc` does not zero.** Any container that grows via `realloc` must explicitly zero the new region, and any struct added to it must be fully initialized.
2. **Fully initialize structs.** Prefer `memset(t, 0, sizeof(*t))` immediately after acquiring the slot, before setting individual fields. A single missing field caused days of debugging.
3. **`calloc` vs `realloc` asymmetry is a classic boundary bug.** The initial 128-slot `calloc` masked the defect for the first 128 tensors; the "first divergence at tensor 130" was the literal realloc boundary.
4. **Buffer size assertions pay off.** A cheap `alloc < needed` check turned a heisenbug into a one-line diagnosis.
5. **Interpreting GPU memory requires matching byte widths.** Early hash comparisons used the wrong element size for the active precision, producing misleading "corruption" conclusions.
6. **ASan cannot see Metal/driver-allocated memory.** For GPU shared buffers, targeted runtime assertions are more effective than generic sanitizers.
7. **Optimization-dependent crashes are a UB smell.** A bug that only faults at `-O2` (vectorized stores) and vanishes at `-O0` is a strong indicator of an out-of-bounds or uninitialized-memory defect.

---

## 8. Reproducing the original bug (historical)

With the pre-`f44c0aa` code, on Apple Silicon:

```sh
cd bare.metal
make baremetal-train
./build/test/g4_test        # flaky: step-0 5.0104 vs 4.9815, occasional NaN
```

The size assertion that identified it:

```sh
# with the temporary BM_CHECK_ADAMW instrumentation
BM_CHECK_ADAMW=1 ./build/test/g4_test 2>&1 | grep "ADAMW OOB"
# ADAMW OOB wid=130 n=884736 alloc=995328 need=1769472
```
