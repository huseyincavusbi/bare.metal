# How Other Engines Handle Model Diversity

Survey of four production LLM inference engines and what they teach us about building a model-agnostic runtime.

## Summary Table

| Engine | Models supported | Per-model files? | Weight format | Model-specific code? | Key pattern |
|--------|-----------------|-----------------|---------------|---------------------|-------------|
| **llama.cpp** | ~25 archs | Yes (1 .cpp each) | GGUF (converted offline) | NO in shared code | Factory + virtual methods |
| **vLLM** | ~150 archs | Yes (1 .py each) | Safetensors (direct) | NO in shared code | Protocol + WeightsMapper |
| **SGLang** | ~212 archs + any HF | Yes (1 .py each) | Safetensors (direct) | In config parser | Auto-discovery + fallback |
| **TensorRT** | Any via ONNX | No | ONNX → compiled engine | Plugin per-op | Graph compiler |

## llama.cpp

**Repo**: `/Users/huseyin/Documents/llmsl/llama.cpp` (582MB, C/C++)

### Architecture

```
GGUF file header: "general.architecture": "gemma"
  → llm_arch_from_string("gemma") → LLM_ARCH_GEMMA enum
  → llama_model_mapping(LLM_ARCH_GEMMA) → new llama_model_gemma()
       ├─ load_arch_hparams()     // read Gemma-specific hyperparams
       ├─ load_arch_tensors()     // create tensors with Gemma naming
       └─ build_arch_graph()      // build Gemma compute graph
           ├─ build_norm(..., LLM_NORM_RMS, ...)
           ├─ build_qkv(...)      // separate Q/K/V (no fused)
           ├─ inpL *= sqrt(n_embd)       ← Gemma quirk
           ├─ Q *= 1/sqrt(head_dim)      ← Gemma quirk
           ├─ build_attn(...)
           └─ build_ffn(..., LLM_FFN_GELU, LLM_FFN_PAR, ...)
```

### Key Design Decisions

1. **Weights converted once at GGUF creation time** (Python). Runtime never transposes or adjusts weights. GPT-2 Conv1D is transposed in the converter, not at load time.

2. **3-method contract** per model: `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`. Each model file only implements what's different; shared code provides parameterized helpers (`build_norm`, `build_qkv`, `build_attn`, `build_ffn`).

3. **Shared helpers are parameterized by enums**, never by if-else on architecture:
   ```cpp
   // Good: parameterized
   cur = build_norm(cur, norm, LLM_NORM_RMS, ctx0, il);
   
   // They never do this:
   if (arch == GEMMA) { /* special case */ }
   ```

4. **Tensor name format strings** with placeholders:
   ```cpp
   { LLM_TENSOR_ATTN_Q, "blk.%d.attn_q" }        // Llama naming
   { LLM_TENSOR_ATTN_Q, "blk.%d.attn_qkv" }      // GPT-2 fused QKV
   ```
   This lets models use different naming conventions without custom code.

5. **Flags-driven variants**: Instead of separate files, some models share code through flags:
   ```cpp
   // llama_model_llama handles Llama, Mistral, InternLM3, Phi3
   // via constructor parameters: layer_type, bias, norm
   ```

### How Quirks Are Handled

| Quirk | Mechanism |
|-------|-----------|
| Gemma RMSNorm+1 | Weight stored as `1+w` at GGUF conversion |
| GPT-2 Conv1D | Transposed at GGUF conversion |
| Embedding scale √D | `ggml_scale(inpL, sqrtf(n_embd))` in graph builder |
| Gated vs sequential FFN | `LLM_FFN_PAR` vs `LLM_FFN_SEQ` enum, handled by `build_ffn()` |
| Attention bias | Loaded with `TENSOR_NOT_REQUIRED` flag when model has no bias |
| Fused QKV (GPT-2, MPT) | `create_tensor_qkv()` tries merged first, falls back to separate |
| QK norm (MPT) | `LLM_TENSOR_ATTN_Q_NORM`/`LLM_TENSOR_ATTN_K_NORM` optional tensors |
| Post-attention norm (Gemma2) | `attn_post_norm` tensor loaded separately |

### Files to study

- `src/llama-arch.cpp:372-607` — tensor name format map (central registry)
- `src/llama-model.cpp:39-313` — model factory (one switch, one-time dispatch)
- `src/llama-graph.cpp:1451-1600` — shared `build_norm()` / `build_qkv()` / `build_ffn()` helpers
- `src/models/gemma.cpp` — example of clean per-model file
- `src/models/gpt2.cpp` — example of fused QKV + bias handling

---

## vLLM

**Repo**: `/Users/huseyin/Documents/llmsl/vllm` (355MB, Python/CUDA)

### Architecture

```
config.json "architectures": ["LlamaForCausalLM"]
  → registry.py → models/llama.py → LlamaForCausalLM
  → AutoWeightsLoader + WeightsMapper → shard & fuse weights
  → model.forward(input_ids, positions, kv_cache)
```

### Key Design Decisions

1. **Protocol-based interfaces** (not ABC inheritance):
   ```python
   @runtime_checkable
   class VllmModel(Protocol):
       def __init__(self, vllm_config, prefix="") -> None: ...
       def embed_input_ids(self, input_ids) -> torch.Tensor: ...
       def forward(self, input_ids, positions) -> T_co: ...
   ```
   Any class with these methods IS a vLLM model — no deep class hierarchy.

2. **Declarative weight mapping** via `WeightsMapper`:
   ```python
   hf_to_vllm_mapper = WeightsMapper(
       orig_to_new_stacked={
           ".q_proj": (".qkv_proj", "q"),     # fuse Q, K, V at load time
           ".k_proj": (".qkv_proj", "k"),
           ".v_proj": (".qkv_proj", "v"),
           ".gate_proj": (".gate_up_proj", 0),  # fuse gate, up at load time
           ".up_proj": (".gate_up_proj", 1),
       }
   )
   ```
   No custom load code — the mapper handles concatenation, sharding, and naming.

3. **Safetensors loaded directly** (no conversion step):
   ```python
   with safe_open(st_file, framework="pt") as f:
       for name in f.keys():
           yield name, f.get_tensor(name)
   ```
   Weights stream from disk directly into model parameters via `default_weight_loader()`.

4. **Quirks handled in model-local code**:
   - Gemma RMSNorm: `GemmaRMSNorm` class (weight starts at zero, `+1.0` at forward)
   - GPT-2 Conv1D: `_transpose_conv1d()` in `load_weights()` override
   - Embed scale: `embed_input_ids()` override per model
   - Activation type: `hidden_act` config key → GeluAndMul mode

5. **Config fixup system**: `MODELS_CONFIG_MAP` maps arch strings to `VerifyAndUpdateConfig` subclasses that fix broken/ambiguous configs before model construction.

### How Quirks Are Handled

| Quirk | Mechanism |
|-------|-----------|
| Gemma RMSNorm+1 | `GemmaRMSNorm` class: `forward(x) = rms_norm(x, self.weight + 1.0)` |
| GPT-2 Conv1D | `loaded_weight = loaded_weight.t()` in `load_weights()` |
| Embedding scale √D | `register_buffer("normalizer", sqrt(D))` in model `__init__` |
| Weight fusion (QKV, gate-up) | `WeightsMapper` declarative remapping |
| Attention bias | Config flag → `attention_bias` constructor param |
| LoRA modules | `packed_modules_mapping` in SupporsLoRA mixin |

### Files to study

- `vllm/model_executor/models/registry.py:71` — model registry (flat dict: arch → (module, class))
- `vllm/model_executor/models/utils.py:46` — `WeightsMapper` and `AutoWeightsLoader`
- `vllm/model_executor/models/llama.py:345` — `hf_to_vllm_mapper` example
- `vllm/model_executor/models/gpt2.py:242` — Conv1D transpose in `load_weights()`
- `vllm/model_executor/layers/layernorm.py:132` — `GemmaRMSNorm` (weight+1 pattern)
- `vllm/model_executor/models/interfaces_base.py:47` — `VllmModel` Protocol definition

---

## SGLang

**Repo**: `/Users/huseyin/Documents/llmsl/sglang` (352MB, Python/CUDA)

### Architecture

```
Any config.json → import_model_classes() auto-discover
  → Native impl found (EntryClass match)? → Use optimized path
  → Not found? → TransformersForCausalLM wraps HuggingFace directly
```

### Key Design Decisions

1. **Auto-discovery via `EntryClass`**: Every model file declares `EntryClass = LlamaForCausalLM` at module level. The registry scans all `.py` files and maps class names → classes. No central registry to maintain when adding a model.

2. **Universal HuggingFace fallback**: `TransformersForCausalLM` wraps the `transformers` library. Every model on HuggingFace works out of the box, just slower (no fused kernels, no SGLang optimizations).

3. **Attention backend per model at runtime**: 38 attention backends (FlashInfer, FlashAttention, Triton, TRT-LLM MLA, CUTLASS MLA, hybrid, etc.) selected automatically based on model architecture and GPU type.

4. **Extensive config parser per model**: 58 files in `configs/` for model-specific config normalization (Qwen3, DeepSeek V4, Falcon H1, etc.)

5. **Serving-oriented design**: RadixAttention KV cache tree for prefix sharing, continuous batching, prefill-decode disaggregation, speculative decoding. Not an inference library — a full serving engine.

### What Makes SGLang Different

- **Radix tree KV cache**: Automatic prefix sharing across concurrent requests (different from simple prefix caching)
- **DSL for structured generation**: `sglang.lang` for composing LLM calls with constrained decoding
- **GPU-GPU disaggregation**: Prefill and decode can run on different GPUs

### Files to study

- `python/sglang/srt/models/registry.py:95` — `import_model_classes()` auto-discovery
- `python/sglang/srt/models/transformers.py` — universal HF fallback
- `python/sglang/srt/model_loader/weight_utils.py:947` — safetensors loading with mmap
- `python/sglang/srt/configs/model_config.py:763` — `_derive_model_shapes()` with per-model branches

---

## TensorRT

**Repo**: `/Users/huseyin/Documents/llmsl/TensorRT` (CUDA-only, not portable to Metal)

### Relevance to baremetal

TensorRT is a **graph compiler**, not an LLM runtime. It treats all models the same: ONNX → optimized engine. However, two patterns are worth stealing:

1. **Plugin system** (`IPluginV2`/`IPluginV3` interface): Self-contained custom ops (kernel + serialization + factory). Each plugin is a directory with `.cu` kernel + `.cpp` plugin class + `CMakeLists.txt`. A `PluginCreatorRegistry` singleton discovers all plugins.

    **For baremetal**: Define `IMetalPlugin` interface for Metal Shading Language kernels following the same pattern.

2. **Graph IR with LLM primitives**: TensorRT 11 added native layer types (`kATTENTION_INPUT`, `kATTENTION_OUTPUT`, `kROTARY_EMBEDDING`, `kKVCACHE_UPDATE`, `kMOE`) — treating LLM building blocks as first-class graph nodes.

    **For baremetal**: Could define a hardware-agnostic graph IR that compiles to Metal kernels, using these same LLM primitives.

3. **Explicit quantization via Q/DQ nodes**: `QUANTIZE`/`DEQUANTIZE` layer types allow mixed-precision computation to be expressed declaratively in the graph.

    **For baremetal**: Same pattern could work with Apple's ANE or Metal's FP16/INT8 support.

### Not worth copying

- CUDA kernel per-SM specialization (Metal GPU is more unified)
- Kernel auto-tuning / tactic selection (Metal shader compilation is fast enough)
- Builder/runtime split (no offline build phase needed for Metal)

---

## Cross-Cutting Patterns

### 1. Weight normalization at load time vs conversion time

| Approach | Used by | Pros | Cons |
|----------|---------|------|------|
| **Conversion time** (Python) | llama.cpp | Simple runtime, clean C code | Requires converter step |
| **Load time** (C) | Our baremetal approach | No converter needed | Loader must handle every quirk |
| **Forward time** (runtime) | vLLM (GemmaRMSNorm) | Model weights stay "canonical" | Slight runtime overhead |

All three approaches work. llama.cpp's conversion-time approach keeps the C runtime simple. Our load-time approach is more ambitious (zero-setup) but harder to get right.

### 2. How the engines scale to 100+ models

- **llama.cpp**: Each model is a small .cpp file. Adding a new model = ~200 lines of code.
- **vLLM**: Each model is a .py file. Adding a new model = ~300 lines.
- **SGLang**: Model files + config parser files. Adding a model = ~400 lines + config parser.

None of them try to handle all models in one file. The per-model file approach scales much better than flags.

### 3. What all engines agree on

- **Don't scatter if-else on architecture** in shared code
- **Use enums/flags** to parameterize shared helpers
- **Weights should flow directly from disk** (no double storage)
- **Quirks belong in model-local code**, not in shared infrastructure
- **Config.json is the source of truth** for architecture detection

---

## What This Means for baremetal

### Our approach (unified flags struct) vs theirs (per-model files)

Our `bm_arch_t` flags approach works for the **3-10 decoder-only transformers** we target. It would break down at ~20+ models or when adding SSMs, VLMs, or encoder-decoders. The per-model file approach scales infinitely but adds complexity.

**Recommendation**: Keep the flags approach for now, but ensure our architecture allows:

1. **Parameterized forward pass** (already done: `bm_arch_t.norm`, `.activation`, `.gated_mlp` etc.)
2. **Weight name format strings** for different naming conventions (not yet done — would fix GPT-2 tensor name mapping)
3. **Plugin-like kernel dispatch** for model-specific ops (QK-norm, post-FFN norm)
4. **Fallback mechanism** — if we can't handle a model, say so clearly rather than producing garbage

### What to add to `bm_arch_t`

Based on every engine's approach:

```
Fields we HAVE:  norm, activation, pos_enc, attention, bias, weight_tie, head_dim,
                 gated_mlp, gemma_norm, embed_scale, weight_layout

Fields we NEED:  qk_norm (for MPT), post_attn_norm (for Gemma2), 
                 post_ffn_norm (for Gemma2), sliding_window (for Gemma3)
```

### What to steal for the ST loader

From llama.cpp's tensor name format strings:
```c
// Instead of hardcoding per-model string patterns:
snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", l);

// Use a lookup:
// llm_tensor_name(tensor_type, layer, suffix) 
// returns the correct string for the detected architecture
```

This would make GPT-2's `h.{l}.attn.c_attn.weight` work without special-casing.

### What NOT to steal

- vLLM's `WeightsMapper` async loading (we're single-GPU, synchronous is fine)
- SGLang's RadixAttention KV cache tree (premature optimization for us)
- TensorRT's kernel auto-tuning (Metal shaders compile fast, no tactic selection needed)
- llama.cpp's GGUF format (we're loading safetensors directly)
- Per-model files (overkill for 3-10 decoder-only transformers)
