"""Convert HuggingFace safetensors to bare.metal checkpoint.
Writes tensors in the EXACT order expected by model.c bmt_model_alloc_buffers."""
import json, os, sys, numpy as np, torch
from safetensors import safe_open

model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/gemma3-270m"
out_path = sys.argv[2] if len(sys.argv) > 2 else "model.bin"

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)

D  = cfg.get('hidden_size', cfg.get('n_embd', 0))
H  = cfg.get('intermediate_size', cfg.get('n_inner', D*4))
L  = cfg.get('num_hidden_layers', cfg.get('n_layer', 0))
NH = cfg.get('num_attention_heads', cfg.get('n_head', 0))
NKV = cfg.get('num_key_value_heads', cfg.get('n_head', NH))
V  = cfg.get('vocab_size', cfg.get('n_vocab', 0))
MS = cfg.get('max_position_embeddings', cfg.get('n_positions', 2048))
HD = cfg.get('head_dim', D // NH)
is_rms = 'rms_norm_eps' in cfg
act = cfg.get('hidden_activation', cfg.get('hidden_act', cfg.get('activation_function', 'gelu_new')))
is_swiglu = act in ('silu', 'swiglu')
model_type = cfg.get('model_type', cfg.get('module', cfg.get('architectures', [''])[0] if isinstance(cfg.get('architectures'), list) else ''))
is_gemma = 'gemma' in model_type.lower() if model_type else False

# Load all tensors (BF16→FP32)
tensors = {}
for fn in sorted(os.listdir(model_dir)):
    if fn.endswith('.safetensors'):
        with safe_open(os.path.join(model_dir, fn), framework='pt') as sf:
            for k in sf.keys():
                t = sf.get_tensor(k).float()
                # Handle fused QKV from GPT-2-style models
                tensors[k] = t.numpy()

def get(*names):
    for n in names:
        if n in tensors:
            t = tensors[n]
            if is_gemma and 'norm.weight' in n and n.endswith('.weight'):
                t = t + 1.0
            return t
    return None

has_qk = any('.q_norm' in k for k in tensors)
has_ffn_post = any('post_feedforward_layernorm' in k for k in tensors)
has_lm_head = any('lm_head' in k for k in tensors)
has_up_proj = any('.mlp.up_proj.weight' in k for k in tensors)
has_combined_qkv = any('.c_attn' in k for k in tensors)
has_wpe = any('wpe' in k for k in tensors)
has_bias = any('.bias' in k for k in tensors)
weight_tie = (not has_lm_head) or has_combined_qkv
is_gpt2 = has_combined_qkv

print(f"D={D} H={H} L={L} NH={NH} NKV={NKV} V={V} HD={HD}")
print(f"RMSNorm={is_rms} SwiGLU={is_swiglu} QK_norm={has_qk} FFN_post={has_ffn_post} weight_tie={weight_tie} gated={has_up_proj} gemma={is_gemma}")

# Header
hdr = [0]*256
hdr[0]=20250718; hdr[1]=1; hdr[2]=0        # fp32
hdr[3]=D; hdr[4]=H; hdr[5]=L; hdr[6]=NH; hdr[7]=NKV
hdr[8]=V; hdr[9]=MS
hdr[10]=1 if is_rms else 0
hdr[11]=1 if is_swiglu else 0  # 0=GELU, 1=SwiGLU
hdr[12]=0 if has_wpe else 1  # pos_enc: learned=0, RoPE=1
hdr[13]=1 if NKV < NH else 0
hdr[14]=1 if has_bias else 0
hdr[15]=1 if weight_tie else 0
hdr[16]=1 if has_qk else 0
hdr[17]=1 if has_ffn_post else 0
hdr[18]=HD
hdr[19]=1 if has_up_proj else 0  # gated_mlp

W = []

# 1. token_embedding_table
W.append(get('model.embed_tokens.weight', 'transformer.wte.weight', 'wte.weight').flatten())

# 2. wpe (learned positional embeddings, e.g. GPT-2)
if has_wpe:
    W.append(get('transformer.wpe.weight', 'model.wpe.weight', 'wpe.weight').flatten())

# 3. ln1w (RMSNorm = one weight per layer, LayerNorm = weight + bias)
for l in range(L):
    W.append(get(f'model.layers.{l}.input_layernorm.weight',
                 f'transformer.h.{l}.ln_1.weight',
                 f'h.{l}.ln_1.weight').flatten())
if not is_rms:
    for l in range(L):
        W.append(get(f'model.layers.{l}.input_layernorm.bias',
                     f'transformer.h.{l}.ln_1.bias',
                     f'h.{l}.ln_1.bias').flatten())

# 4. qw: [L][NH*HD][D]
if is_gpt2:
    for l in range(L):
        ca = get(f'transformer.h.{l}.attn.c_attn.weight', f'h.{l}.attn.c_attn.weight')
        W.append(ca[:,:D].T.flatten())  # Q [D, D] → transpose → [768, 768]
    for l in range(L):
        ca = get(f'transformer.h.{l}.attn.c_attn.weight', f'h.{l}.attn.c_attn.weight')
        W.append(ca[:,D:2*D].T.flatten())  # K
    for l in range(L):
        ca = get(f'transformer.h.{l}.attn.c_attn.weight', f'h.{l}.attn.c_attn.weight')
        W.append(ca[:,2*D:].T.flatten())  # V
else:
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.q_proj.weight',
                     f'transformer.h.{l}.attn.c_attn.weight').flatten())
    # 5. kw: [L][NKV*HD][D]
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.k_proj.weight').flatten())
    # 6. vw: [L][NKV*HD][D]
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.v_proj.weight').flatten())

# 7. qkvb (QKV bias)
if has_bias:
    if is_gpt2:
        for l in range(L):
            W.append(get(f'transformer.h.{l}.attn.c_attn.bias', 
                        f'h.{l}.attn.c_attn.bias').flatten())
    else:
        for l in range(L):
            W.append(get(f'model.layers.{l}.self_attn.q_proj.bias').flatten())
        for l in range(L):
            W.append(get(f'model.layers.{l}.self_attn.k_proj.bias').flatten())
        for l in range(L):
            W.append(get(f'model.layers.{l}.self_attn.v_proj.bias').flatten())

# 8. q_norm_w (if has_qk_norm)
if has_qk:
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.q_norm.weight').flatten())
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.k_norm.weight').flatten())

# 9. attprojw: [L][NH*HD][D] + attprojb
if is_gpt2:
    for l in range(L):
        W.append(get(f'transformer.h.{l}.attn.c_proj.weight',
                     f'h.{l}.attn.c_proj.weight').T.flatten())
else:
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.o_proj.weight',
                     f'transformer.h.{l}.attn.c_proj.weight').flatten())
if has_bias:
    for l in range(L):
        W.append(get(f'model.layers.{l}.self_attn.o_proj.bias',
                     f'transformer.h.{l}.attn.c_proj.bias',
                     f'h.{l}.attn.c_proj.bias').flatten())

# 10. ln2w: [L][D]
for l in range(L):
    W.append(get(f'model.layers.{l}.post_attention_layernorm.weight',
                 f'transformer.h.{l}.ln_2.weight',
                 f'h.{l}.ln_2.weight').flatten())
if not is_rms:
    for l in range(L):
        W.append(get(f'model.layers.{l}.post_attention_layernorm.bias',
                     f'transformer.h.{l}.ln_2.bias',
                     f'h.{l}.ln_2.bias').flatten())

# 11. fcw (w1/gate): [L][H][D] + fcb
if is_gpt2:
    for l in range(L):
        W.append(get(f'transformer.h.{l}.mlp.c_fc.weight',
                     f'h.{l}.mlp.c_fc.weight').T.flatten())
else:
    for l in range(L):
        W.append(get(f'model.layers.{l}.mlp.gate_proj.weight',
                     f'transformer.h.{l}.mlp.c_fc.weight',
                     f'h.{l}.mlp.c_fc.weight').flatten())
if has_bias:
    for l in range(L):
        W.append(get(f'model.layers.{l}.mlp.gate_proj.bias',
                     f'transformer.h.{l}.mlp.c_fc.bias',
                     f'h.{l}.mlp.c_fc.bias').flatten())

# 12. fcw3 (w3/up, for gated FFN)
if has_up_proj:
    for l in range(L):
        W.append(get(f'model.layers.{l}.mlp.up_proj.weight').flatten())

# 13. fcprojw (w2/down): [L][D][H] + fcprojb
if is_gpt2:
    for l in range(L):
        W.append(get(f'transformer.h.{l}.mlp.c_proj.weight',
                     f'h.{l}.mlp.c_proj.weight').T.flatten())
else:
    for l in range(L):
        W.append(get(f'model.layers.{l}.mlp.down_proj.weight',
                     f'transformer.h.{l}.mlp.c_proj.weight',
                     f'h.{l}.mlp.c_proj.weight').flatten())
if has_bias:
    for l in range(L):
        W.append(get(f'model.layers.{l}.mlp.down_proj.bias',
                     f'transformer.h.{l}.mlp.c_proj.bias',
                     f'h.{l}.mlp.c_proj.bias').flatten())

# 14. pre_ffn_w + ffn_post_w (if has_ffn_post_norm)
if has_ffn_post:
    for l in range(L):
        W.append(get(f'model.layers.{l}.pre_feedforward_layernorm.weight').flatten())
    for l in range(L):
        W.append(get(f'model.layers.{l}.post_feedforward_layernorm.weight').flatten())

# 15. lnfw: [D] + lnfb
W.append(get('model.norm.weight', 'transformer.ln_f.weight', 'ln_f.weight').flatten())
if has_bias:
    W.append(get('model.norm.bias', 'transformer.ln_f.bias', 'ln_f.bias').flatten())

# 16. wcls (if no weight tie)
if not weight_tie:
    lm = get('lm_head.weight')
    if lm is None:
        lm = get('model.embed_tokens.weight')
    W.append(lm.flatten())

total = sum(w.size for w in W)
with open(out_path, 'wb') as f:
    f.write(np.array(hdr, dtype=np.int32).tobytes())
    for w in W:
        f.write(w.astype(np.float32).tobytes())

print(f"Wrote {out_path}: {total} params ({total*4} bytes)")
print(f"Expected arch: norm={'RMSNorm' if is_rms else 'LayerNorm'} act={'SwiGLU' if is_swiglu else 'GELU'}")
