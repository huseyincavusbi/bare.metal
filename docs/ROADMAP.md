# Philosophy & Roadmap

## What changed after studying other engines

### Confirmed: our approach is correct

Every engine uses flags/enums/parameters to describe model differences. They just organize them differently:

| Engine | How they encode differences | How they dispatch |
|--------|---------------------------|-------------------|
| **llama.cpp** | Enums (`LLM_NORM_RMS`, `LLM_FFN_GELU`) per model file | Factory → virtual method |
| **vLLM** | Config keys (`hidden_act`, `rms_norm_eps`) per model file | Registry → Protocol |
| **SGLang** | Config + per-model class | EntryClass auto-discovery |
| **baremetal** | Single `bm_arch_t` flags struct | **One forward pass, checks flags at runtime** |

They all do the same thing under the hood — the difference is organization, not concept. They use per-model files because they have **200+ models**. We have **3-10**. For our scope, a unified forward pass with flag checks is simpler and correct.

### What we do NOT need to copy

- Per-model files (overkill for 3-10 decoder-only transformers)
- Multi-GPU tensor/pipeline parallelism (single Apple Silicon GPU)
- Radix tree KV caching, prefill-decode disaggregation (serving infra)
- Kernel auto-tuning / tactic selection (Metal shaders compile fast)
- GGUF format (we load safetensors directly)
- Python converter step (we adjust weights at C load time)
- Attention backend selection (one Metal backend)

### What we SHOULD steal

1. **Tensor name format strings** (llama.cpp) — instead of hardcoding model-specific paths in the ST loader, use a lookup table:
   ```
   T_ATTN_Q → "model.layers.%d.self_attn.q_proj.weight" (Llama)
   T_ATTN_Q → "h.%d.attn.c_attn.weight" (GPT-2, fused QKV)
   ```

2. **Parameterized shared helpers** (llama.cpp) — our `R`/`L`/`N` macros already do this. Forward pass calls `N(x, w, b)` and the macro dispatches to RMSNorm or LayerNorm based on `nm`. Same pattern as llama.cpp's `build_norm(cur, norm, LLM_NORM_RMS)`.

3. **Safetensors direct loading** (vLLM, SGLang) — we already do this in `checkpoint_st.c`. Just need to add the weight-adjustment logic.

4. **LLM primitives as first-class ops** (TensorRT) — we already have this in `forward.metal` kernels. QK-norm, RoPE, RMSNorm, LayerNorm, GELU, SwiGLU, all separate Metal functions.

## The philosophy

**Old**: One struct of flags + one allocator + one forward pass. The flags define everything.

**New** (same thing, validated): One struct of flags + one allocator + one forward pass. The flags drive dispatch at every step. Zero manual configuration. The ST loader reads config.json + tensor names → auto-fills flags. No converter. No per-model files. No scattered if-else (just parameterized macros). This genuinely works for 3-10 decoder-only transformers.

The difference from llama.cpp/vLLM is **not** that we handle models differently — we handle them the **same way** (flags/enums/parameters). The difference is we put it all in **one struct, one file** instead of N files with N virtual methods. For our model count, this is an advantage, not a limitation.

## Concrete roadmap

### Phase 1: Fix what's broken (now)

| # | Task | Why |
|---|------|-----|
| 1 | Fix SV matmul transpose in attention | This is the "always garbage" bug. QK^T matmul uses `[S][HD]` layout, SV matmul also needs `[S][HD]` but kernel reads `[HD][S]`. Only matches when S=1 (first token). All subsequent tokens get corrupted attention output. |
| 2 | Add CPT_NORM macro to ST loader | Gemma uses `(1+w)` RMSNorm. Add `+1.0` at load time so runtime sees normal RMSNorm weights. |
| 3 | Add CPT_T macro to ST loader | GPT-2 uses Conv1D `[in,out]` layout. Transpose to `[out,in]` at load time so runtime sees normal matmul layout. |
| 4 | Add tensor name format lookup | Instead of hardcoded `"model.layers.%d.self_attn.q_proj.weight"`, use a format table so GPT-2's `"h.%d.attn.c_attn.weight"` and Gemma's `"model.layers.%d.self_attn.q_proj.weight"` both resolve correctly. |
| 5 | Fix 1.77M param allocation mismatch | ST loader allocates 136M params but converter allocates 134M. Root cause: some arch flag differs between ST detection and converter header. Debug by comparing `bm_arch_t` field-by-field. |

### Phase 2: Validate all models

| # | Task | Target |
|---|------|--------|
| 6 | Validate SmolLM2 end-to-end | cos>0.70, generation is coherent |
| 7 | Validate Gemma3 | cos>0.70 (RMSNorm+1, embed_scale, gated GELU) |
| 8 | Validate GPT-2 | cos>0.70 (Conv1D, LayerNorm, learned pos, fused QKV) |

### Phase 3: Clean up

| # | Task |
|---|------|
| 9 | Delete `scripts/convert.py` — ST loader is the only path |
| 10 | Delete `scripts/convert_hf.py` — redundant |
| 11 | Remove legacy binary checkpoint loader from `api.c` (keep `checkpoint.c` for training checkpoints) |

### Phase 4: Beyond 3 models (if needed)

If we ever need 20+ models or non-decoder architectures (SSM, VLM, encoder-decoder):

```
bm_arch_t flags → bm_get_forward_fn() → specialized forward function
```

Each model type gets its own `run_llama.c`, `run_gemma.c`, etc. But the allocator, loader, and kernel selection stay unified. This is llama.cpp's factory pattern — but only when we need it.

## What runs today

| Model | Format | Status |
|-------|--------|--------|
| **SmolLM2-135M** | Binary checkpoint (converter) | Loads, cos was ~0.78 on first token, generation is garbage due to SV matmul bug |
| **SmolLM2-135M** | ST loader (direct safetensors) | Loads, 1.77M param mismatch, generation garbage |
| **Gemma3-270M** | Binary checkpoint (converter) | Loads, cos was ~0.30 (RMSNorm+1 etc not fully fixed) |
| **Gemma3-270M** | ST loader | Not tested yet |
| **GPT-2 124M** | Binary checkpoint (converter) | Broke (layernorm kernel + Conv1D) |
| **GPT-2 124M** | ST loader | Not tested yet |

One bug (SV matmul) affects all models equally. Fix that first.
