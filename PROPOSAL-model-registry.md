# Proposal: Pluggable Model Definitions (Model-Family Registry)

**Goal:** make `bare.metal` agnostic across decoder-only transformer LLMs (Llama, Mistral,
Qwen2, Gemma, Phi-3, GPT-2/GPT-J, Falcon, MPT, …) without touching the kernel library or
the scheduler. Support grows additively by adding a small model-definition module per family.

**Non-goals:** general tensor compilers (ONNX Runtime / TVM / XLA territory), arbitrary
topologies, MoE/SSM/multimodal, encoder-decoder. Those are out of scope by design.

---

## 1. Why the engine is not agnostic today

The runtime (graph IR, buffer planner, Metal scheduler, kernels) is already generic. The
model-specific assumptions are concentrated in two places:

1. **Weight binding** — `src/baremetal/checkpoint_st.c` hardcodes HuggingFace-Llama tensor
   names:
   ```
   model.embed_tokens.weight
   model.layers.N.input_layernorm.weight
   model.layers.N.self_attn.{q,k,v,o}_proj.weight
   model.layers.N.post_attention_layernorm.weight
   model.layers.N.mlp.{gate,up,down}_proj.weight
   model.norm.weight
   ```
   GPT-2 (`transformer.h.N.attn.c_attn`), Falcon (`transformer.h.N.self_attention.query_key_value`),
   MPT (`transformer.blocks.N.attn.Wqkv`), etc. never match.

2. **Graph construction** — `src/baremetal/graph.c` (`bmt_graph_build`,
   `bmt_graph_build_train`) emits one fixed pre-norm decoder block
   (`norm → attn → add → norm → mlp → add`) with inline `if (arch.x)` branches.

The `bm_arch_t` descriptor already captures most *numeric* variation (`dim`, `hidden_dim`,
`n_layers`, `n_heads`, `n_kv_heads`, `norm`, `activation`, `pos_enc`, `attention`, `bias`,
`weight_tie`, `gated_mlp`, `gemma_norm`, `embed_scale`, `has_qk_norm`, `has_ffn_post_norm`,
`head_dim`, `weight_layout`, `rope_theta`) and is consumed by the graph builder. So the
descriptor abstraction exists; what's missing is a **model-family layer** that (a) knows the
name mapping and (b) knows the topology.

---

## 2. Design: three layers

```
┌───────────────────────────────────────────────────────────────┐
│  Model-definition registry   (NEW)                            │
│    llama_def, gpt2_def, gemma_def, mistral_def, qwen2_def …    │
│    each: matches() / bind() / build() (+ optional tokenizer)   │
├───────────────────────────────────────────────────────────────┤
│  Runtime: graph IR + scheduler + buffers + kernels  (EXISTS)   │
│    generic ops: matmul, rmsnorm/layernorm, rope, attention,    │
│    swiglu/gelu, add; CPU/Metal backends; fp32/bf16/fp16/q8     │
├───────────────────────────────────────────────────────────────┤
│  Formats / IO                      (EXISTS, to generalize)     │
│    safetensors + .bin loader, BPE tokenizer.json               │
└───────────────────────────────────────────────────────────────┘
```

The engine stays generic; **only layer 1 becomes pluggable**.

---

## 3. Core abstraction 1 — role-based weight table

Replace the long list of hardcoded pointers in `bm_model_s`
(`qw, kw, vw, attprojw, fcw, fcw3, fcprojw, ln1w, ln2w, lnfw, …`) with a
`(role, layer)` table. Role is family-independent; each family's `bind()` fills it.

```c
// include/baremetal/roles.h
typedef enum {
    BM_ROLE_EMBED,          // token embedding               [V, D]
    BM_ROLE_POS_EMBED,      // learned positional embedding  [Smax, D]  (optional)
    BM_ROLE_LM_HEAD,        // output projection (tied or not)[V, D]
    BM_ROLE_FINAL_NORM,     // final norm weight             [D]

    BM_ROLE_ATTN_Q,         // (fused qkv families leave these NULL)
    BM_ROLE_ATTN_K,
    BM_ROLE_ATTN_V,
    BM_ROLE_ATTN_QKV,       // fused QKV (GPT-2 c_attn, Falcon, MPT)
    BM_ROLE_ATTN_O,
    BM_ROLE_ATTN_Q_NORM,    // QK-norm (Qwen2, Gemma2)       [HD]
    BM_ROLE_ATTN_K_NORM,

    BM_ROLE_MLP_UP,         // [H, D]  (gated)
    BM_ROLE_MLP_GATE,       // [H, D]  (gated)
    BM_ROLE_MLP_DOWN,       // [D, H]
    BM_ROLE_MLP_FC,         // [H, D]  (non-gated GELU/ReLU)

    BM_ROLE_NORM1,          // pre-attention norm            [D]
    BM_ROLE_NORM2,          // pre-MLP norm                  [D]
    BM_ROLE_MLP_PRE_NORM,   // Gemma-style extra norms       [D]
    BM_ROLE_MLP_POST_NORM,

    BM_ROLE_NORM1_BIAS,     // LayerNorm bias (GPT-2 etc.)   [D]
    BM_ROLE_NORM2_BIAS,
    BM_ROLE_FINAL_NORM_BIAS,
    BM_ROLE_QKV_BIAS,       // attention bias
    BM_ROLE_MLP_BIAS,       // optional

    BM_ROLE_COUNT
} bm_weight_role_t;

#define BM_MAX_LAYERS 128

// in struct bm_model_s
typedef struct bm_model_s {
    bm_arch_t arch;
    bm_precision_t precision;
    size_t n_parameters;

    void*  weight_buffer;                       // fp32 master allocation
    void*  weights[BM_ROLE_COUNT][BM_MAX_LAYERS]; // [role][layer] -> into weight_buffer
    int    layer_count[BM_ROLE_COUNT];            // values: 0, 1, or n_layers

    const struct bm_model_def_s* def;           // owning family
    void*  graph;
    int    quantized;
    int    kv_dim, head_size, n_kv_heads, kv_mul;
} bm_model_t;

static inline void* bm_weight(bm_model_t* m, bm_weight_role_t r, int layer) {
    return m->weights[r][layer];
}
```

This single change removes every family-specific pointer name from the engine core. The
existing `bmt_graph_build*` code is rewritten once in terms of `bm_weight(...)`.

---

## 4. Core abstraction 2 — model-definition vtable

```c
// include/baremetal/model_def.h
typedef struct {
    const char* name;                 // "LlamaForCausalLM", "GPT2LMHeadModel", …
    const char* family;               // "llama", "gpt2", …

    /* 1. Does this checkpoint belong to this family?
     *    Inspect config.json text + safetensors tensor names. */
    int (*matches)(const char* config_json,
                   const char* const* tensor_names, int n_tensors);

    /* 2. Fill bm_arch_t and bind every weight slot into the (role, layer) table.
     *    Uses the generic safetensors accessor; family-specific names live here. */
    int (*bind)(bm_model_t* m, bm_arch_t* arch,
                const bm_st_file_t* const* files, int n_files);

    /* 3. Optional: emit the compute graph. If NULL, the default standard
     *    pre-norm decoder builder is used (covers Llama/Mistral/Qwen/Gemma). */
    int (*build)(bm_model_t* m, bmt_graph_t* g, int S, int training);

    /* 4. Optional: tokenizer format hint ("bpe_json", "sentencepiece"). */
    const char* tokenizer_format;
} bm_model_def_t;

/* registry */
const bm_model_def_t* bm_model_detect(const char* config_json,
                                      const char* const* tensor_names,
                                      int n_tensors);
int  bm_model_register(const bm_model_def_t* def);   // for out-of-tree families
```

Detection order matters (specific → general), e.g. `gemma` before `llama` (both share
`LlamaForCausalLM`-ish names), `qwen2` before `llama` (QK-norm + bias).

---

## 5. Default builders for the common block

Most families differ only in *flags*, not topology. Provide two reusable builders:

```c
/* Standard sequential pre-norm block:
 *   x + attn(norm1(x));  x + mlp(norm2(x))
 * Handles RMSNorm/LayerNorm, MHA/GQA, RoPE/learned, gated/plain MLP, bias, QK-norm. */
int bm_build_decoder_sequential(bm_model_t* m, bmt_graph_t* g, int S, int training,
                                const bm_block_opts_t* opts);

/* Parallel-residual block (GPT-J / PaLM / Falcon):
 *   x + attn(norm1(x)) + mlp(norm2(x))   (attn and MLP read the same normed input) */
int bm_build_decoder_parallel(bm_model_t* m, bmt_graph_t* g, int S, int training);

typedef struct {
    int parallel_residual;   // 0 or 1
    int norm_before_mlp;     // pre-norm vs post-norm placement
    int qk_norm;
    int mlp_bias;
    int shared_embed;        // tied lm_head
} bm_block_opts_t;
```

`build` in the vtable is then rarely needed:
- **Llama / Mistral / Qwen2 / SmolLM2 / TinyLlama / Phi-3** → `build = NULL`,
  use `sequential` with flags.
- **Gemma / Gemma2** → `sequential` + `gemma_norm` (extra pre/post-FFN norms) + `embed_scale`.
- **GPT-2 / GPT-J / Falcon / MPT** → custom `build` (parallel residual, fused QKV, learned pos,
  LayerNorm+bias) or `parallel` where it fits.

---

## 6. Example: a family definition

```c
// src/models/llama.c
static int llama_matches(const char* cfg, const char* const* names, int n) {
    if (strstr(cfg, "\"model_type\": \"llama\"")) return 1;
    if (strstr(cfg, "LlamaForCausalLM")) return 1;
    for (int i = 0; i < n; i++)                       // overlap with HF Llama names
        if (strstr(names[i], "self_attn.q_proj")) return 1;
    return 0;
}

static int llama_bind(bm_model_t* m, bm_arch_t* a,
                      const bm_st_file_t* const* files, int nf) {
    llama_fill_arch(a, /*config*/ ...);               // dim, heads, rope_theta, GQA, …
    float* w = m->weight_buffer;
    #define ROLE(role, layer, name, count) \
        bm_st_bind(files, nf, name, bm_weight(m, role, layer), count)
    ROLE(BM_ROLE_EMBED, 0, "model.embed_tokens.weight", V*D);
    ROLE(BM_ROLE_FINAL_NORM, 0, "model.norm.weight", D);
    for (int l = 0; l < a->n_layers; l++) {
        char b[256];
        #define L(name) (snprintf(b, sizeof b, name, l), b)
        ROLE(BM_ROLE_NORM1,        l, L("model.layers.%d.input_layernorm.weight"), D);
        ROLE(BM_ROLE_NORM2,        l, L("model.layers.%d.post_attention_layernorm.weight"), D);
        ROLE(BM_ROLE_ATTN_Q,       l, L("model.layers.%d.self_attn.q_proj.weight"), a->dim*NH*HD);
        ROLE(BM_ROLE_ATTN_K,       l, L("model.layers.%d.self_attn.k_proj.weight"), a->dim*KV);
        ROLE(BM_ROLE_ATTN_V,       l, L("model.layers.%d.self_attn.v_proj.weight"), a->dim*KV);
        ROLE(BM_ROLE_ATTN_O,       l, L("model.layers.%d.self_attn.o_proj.weight"), NH*HD*a->dim);
        ROLE(BM_ROLE_ATTN_Q_NORM,  l, L("model.layers.%d.self_attn.q_norm.weight"), HD); // Qwen2
        ROLE(BM_ROLE_ATTN_K_NORM,  l, L("model.layers.%d.self_attn.k_norm.weight"), HD);
        ROLE(BM_ROLE_MLP_GATE,     l, L("model.layers.%d.mlp.gate_proj.weight"), a->hidden_dim*a->dim);
        ROLE(BM_ROLE_MLP_UP,       l, L("model.layers.%d.mlp.up_proj.weight"),   a->hidden_dim*a->dim);
        ROLE(BM_ROLE_MLP_DOWN,     l, L("model.layers.%d.mlp.down_proj.weight"), a->dim*a->hidden_dim);
        ROLE(BM_ROLE_MLP_FC,       l, L("model.layers.%d.mlp.fc1.weight"),       a->hidden_dim*a->dim); // Phi-2
    }
    return 0;
}

const bm_model_def_t bm_def_llama = {
    .name = "LlamaForCausalLM",
    .family = "llama",
    .matches = llama_matches,
    .bind = llama_bind,
    .build = NULL,                        // use default sequential builder
    .tokenizer_format = "bpe_json",
};
```

```c
// src/models/gpt2.c  (custom topology)
static int gpt2_matches(const char* cfg, const char* const* names, int n) {
    if (strstr(cfg, "\"model_type\": \"gpt2\"")) return 1;
    for (int i = 0; i < n; i++)
        if (strstr(names[i], "transformer.h.")) return 1;
    return 0;
}

static int gpt2_bind(bm_model_t* m, bm_arch_t* a, const bm_st_file_t* const* f, int nf) {
    gpt2_fill_arch(a);                    // learned pos, LayerNorm, GELU, MHA, bias
    #define ROLE(role, layer, name, count) bm_st_bind(f, nf, name, bm_weight(m, role, layer), count)
    ROLE(BM_ROLE_EMBED,    0, "transformer.wte.weight", V*D);
    ROLE(BM_ROLE_POS_EMBED,0, "transformer.wpe.weight", a->max_seq_len*D);
    ROLE(BM_ROLE_FINAL_NORM,0,"transformer.ln_f.weight", D);
    for (int l = 0; l < a->n_layers; l++) {
        char b[256];
        #define L(name) (snprintf(b, sizeof b, name, l), b)
        ROLE(BM_ROLE_NORM1,       l, L("transformer.h.%d.ln_1.weight"), D);
        ROLE(BM_ROLE_NORM1_BIAS,  l, L("transformer.h.%d.ln_1.bias"),   D);
        ROLE(BM_ROLE_ATTN_QKV,    l, L("transformer.h.%d.attn.c_attn.weight"), 3*D*D); // fused
        ROLE(BM_ROLE_QKV_BIAS,    l, L("transformer.h.%d.attn.c_attn.bias"),   3*D);
        ROLE(BM_ROLE_ATTN_O,      l, L("transformer.h.%d.attn.c_proj.weight"), D*D);
        ROLE(BM_ROLE_NORM2,       l, L("transformer.h.%d.ln_2.weight"), D);
        ROLE(BM_ROLE_NORM2_BIAS,  l, L("transformer.h.%d.ln_2.bias"),   D);
        ROLE(BM_ROLE_MLP_FC,      l, L("transformer.h.%d.mlp.c_fc.weight"),  H*D);
        ROLE(BM_ROLE_MLP_DOWN,    l, L("transformer.h.%d.mlp.c_proj.weight"), D*H);
    }
    return 0;
}

const bm_model_def_t bm_def_gpt2 = {
    .name = "GPT2LMHeadModel", .family = "gpt2",
    .matches = gpt2_matches, .bind = gpt2_bind,
    .build = bm_build_gpt2,               // parallel residual + fused QKV + learned pos
    .tokenizer_format = "bpe_json",
};
```

Registration is a static table; no dynamic loading required:

```c
// src/models/registry.c
static const bm_model_def_t* g_defs[] = {
    &bm_def_gemma, &bm_def_qwen2, &bm_def_gpt2, &bm_def_falcon, &bm_def_llama,
};
const bm_model_def_t* bm_model_detect(const char* cfg, const char* const* names, int n) {
    for (size_t i = 0; i < sizeof g_defs/sizeof *g_defs; i++)
        if (g_defs[i]->matches(cfg, names, n)) return g_defs[i];
    return NULL;
}
```

---

## 7. Migration path (incremental, no big bang)

| Step | Change | Risk |
|---|---|---|
| 1 | Add `roles.h` + `weights[role][layer]` to `bm_model_s`; keep old named pointers as aliases. | none |
| 2 | Extract the current Llama name mapping from `checkpoint_st.c` into `src/models/llama.c`. Behavior identical. | low |
| 3 | Rewrite `bmt_graph_build*` to read via `bm_weight(...)`; move the Llama block into `bm_build_decoder_sequential`. | medium |
| 4 | Add `bm_model_detect` + registry; call it from `bm_load_weights`. | low |
| 5 | Add `src/models/gpt2.c` (+ parallel builder) as the first proof of a non-Llama family. | medium |
| 6 | Add Qwen2 (QK-norm), Gemma/2 (extra norms + embed scale), Mistral, Phi-3 (mostly flags). | low each |

Steps 1–4 are a pure refactor with no behavior change; new families are then additive.

**Testing:** each family gets a `gN_test` that compares a forward against a PyTorch/transformers
reference for the same checkpoint (the same golden-ref pattern already used for g3).

---

## 8. Effort & scope

| Deliverable | Estimate |
|---|---|
| Roles + `bm_weight` + loader refactor (steps 1–4) | ~2–3 days |
| Default sequential builder abstraction (step 3) | ~2 days |
| First new family end-to-end (GPT-2, parallel builder) | ~2–3 days |
| Qwen2 / Gemma / Mistral / Phi-3 (flag-only, with refs) | ~0.5–1 day each |
| SentencePiece tokenizer support | ~1–2 days |

**Result:** out-of-the-box support for the HF decoder-only ecosystem (Llama, SmolLM2,
TinyLlama, Mistral, Qwen2/2.5, Gemma/2, Phi-3, GPT-2, GPT-J, Falcon, MPT), with new
families added as isolated modules that share the kernel library and scheduler unchanged.

---

## 9. What remains out of scope (by design)

- MoE routing (Mixtral, Qwen-MoE), SSM/hybrid (Mamba, Jamba), sliding-window/ALiBi, attention
  sinks, cross-attention, encoder-decoder, multimodal.
- These require new *ops/topologies*, not just binding. If ever needed, they slot into the
  same layer-1 registry as modules that also contribute new kernel ops — but they are
  deliberately not part of this proposal.

---

## 10. One-line summary

> Keep the kernel library and scheduler generic; introduce a **role-based weight table** and a
> **model-definition registry** (matches/bind/build), with **default block builders** so most
> families need only a ~100-line name-mapping module. This turns "SmolLM-specific" into
> "agnostic across the HF decoder-only transformer family," additively and without a rewrite.