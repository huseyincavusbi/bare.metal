import struct, json, sys, os, numpy as np, torch
from safetensors import safe_open

def convert_hf_to_baremetal(model_dir, output_path):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    arch = {}
    arch['dim'] = cfg.get('hidden_size', cfg.get('d_model', cfg.get('n_embd')))
    arch['hidden_dim'] = cfg.get('intermediate_size', cfg.get('n_inner',
        4 * arch['dim']))
    arch['n_layers'] = cfg.get('num_hidden_layers', cfg.get('n_layer'))
    arch['n_heads'] = cfg.get('num_attention_heads', cfg.get('n_head'))
    arch['n_kv_heads'] = cfg.get('num_key_value_heads', arch['n_heads'])
    arch['vocab_size'] = cfg.get('vocab_size')
    arch['max_seq_len'] = cfg.get('max_position_embeddings',
                            cfg.get('n_positions', cfg.get('n_ctx', 1024)))

    # Architecture detection
    has_rope = cfg.get('rope_theta') is not None or cfg.get('rope_scaling') is not None
    arch['norm'] = 1 if 'rms' in str(cfg.get('norm_type', '')).lower() else 0
    act = cfg.get('hidden_act', 'gelu_new').lower()
    arch['activation'] = 1 if act in ('silu', 'swiglu') else 0
    arch['pos_enc'] = 1 if has_rope else 0
    arch['attention'] = 1 if arch['n_kv_heads'] < arch['n_heads'] else 0
    has_bias = cfg.get('bias', True) or cfg.get('attention_bias', True)
    arch['bias'] = 1 if has_bias else 0
    arch['weight_tie'] = 1 if cfg.get('tie_word_embeddings',
                         cfg.get('tie_weights', True)) else 0

    L, D, H, V = arch['n_layers'], arch['dim'], arch['hidden_dim'], arch['vocab_size']
    NH, HD = arch['n_heads'], D // arch['n_heads']

    header = [0] * 256
    header[0] = 20250718; header[1] = 1; header[2] = 0
    header[3] = D; header[4] = H; header[5] = L; header[6] = NH
    header[7] = arch['n_kv_heads']; header[8] = V; header[9] = arch['max_seq_len']
    header[10] = arch['norm']; header[11] = arch['activation']
    header[12] = arch['pos_enc']; header[13] = arch['attention']
    header[14] = arch['bias']; header[15] = arch['weight_tie']

    tensors = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith('.safetensors'):
            with safe_open(os.path.join(model_dir, fn), framework='pt') as sf:
                for k in sf.keys():
                    tensors[k] = sf.get_tensor(k).float().numpy()

    def get(*names):
        for n in names:
            if n in tensors: return tensors[n]
        return None

    weights = []
    wte = get('model.embed_tokens.weight', 'transformer.wte.weight', 'wte.weight')
    if wte is None: wte = get(f'model.layers.0.self_attn.q_proj.weight'); wte = np.zeros((V, D))
    weights.append(wte.flatten())

    if arch['pos_enc'] == 0:
        wpe = get('transformer.wpe.weight', 'wpe.weight')
        if wpe is not None: weights.append(wpe.flatten())

    is_gpt2 = 'transformer' in str(tensors.keys())

    for l in range(L):
        ln1 = get(f'model.layers.{l}.input_layernorm.weight',
                  f'transformer.h.{l}.ln_1.weight',
                  f'h.{l}.ln_1.weight')
        if ln1 is not None: weights.append(ln1.flatten())
        if arch['norm'] == 0:  # LayerNorm has bias
            ln1b = get(f'model.layers.{l}.input_layernorm.bias',
                       f'transformer.h.{l}.ln_1.bias',
                       f'h.{l}.ln_1.bias')
            if ln1b is not None: weights.append(ln1b.flatten())

        q = get(f'model.layers.{l}.self_attn.q_proj.weight',
                f'transformer.h.{l}.attn.c_attn.weight',
                f'h.{l}.attn.c_attn.weight')
        k = get(f'model.layers.{l}.self_attn.k_proj.weight')
        v = get(f'model.layers.{l}.self_attn.v_proj.weight')

        if q is not None and k is not None and v is not None:
            # Llama-style: separate Q, K, V
            qkv = np.concatenate([q, k, v], axis=0)
        elif q is not None:
            # GPT-2 style: fused QKV
            qkv = q
        else:
            qkv = np.zeros((3 * NH * HD, D))
        weights.append(qkv.flatten())

        if arch['bias']:
            qb = get(f'model.layers.{l}.self_attn.q_proj.bias',
                     f'transformer.h.{l}.attn.c_attn.bias')
            if qb is not None: weights.append(qb.flatten())

        o = get(f'model.layers.{l}.self_attn.o_proj.weight',
                f'transformer.h.{l}.attn.c_proj.weight',
                f'h.{l}.attn.c_proj.weight')
        if o is not None: weights.append(o.flatten())
        if arch['bias']:
            ob = get(f'model.layers.{l}.self_attn.o_proj.bias',
                     f'transformer.h.{l}.attn.c_proj.bias')
            if ob is not None: weights.append(ob.flatten())

        ln2 = get(f'model.layers.{l}.post_attention_layernorm.weight',
                  f'transformer.h.{l}.ln_2.weight',
                  f'h.{l}.ln_2.weight')
        if ln2 is not None: weights.append(ln2.flatten())
        if arch['norm'] == 0:
            ln2b = get(f'model.layers.{l}.post_attention_layernorm.bias',
                       f'transformer.h.{l}.ln_2.bias')
            if ln2b is not None: weights.append(ln2b.flatten())

        w1 = get(f'model.layers.{l}.mlp.gate_proj.weight',
                 f'transformer.h.{l}.mlp.c_fc.weight',
                 f'h.{l}.mlp.c_fc.weight')
        if w1 is not None: weights.append(w1.flatten())
        if arch['bias']:
            w1b = get(f'model.layers.{l}.mlp.gate_proj.bias',
                      f'transformer.h.{l}.mlp.c_fc.bias')
            if w1b is not None: weights.append(w1b.flatten())

        if arch['activation'] == 1:
            w3 = get(f'model.layers.{l}.mlp.up_proj.weight')
            if w3 is not None: weights.append(w3.flatten())

        w2 = get(f'model.layers.{l}.mlp.down_proj.weight',
                 f'transformer.h.{l}.mlp.c_proj.weight',
                 f'h.{l}.mlp.c_proj.weight')
        if w2 is not None: weights.append(w2.flatten())
        if arch['bias']:
            w2b = get(f'model.layers.{l}.mlp.down_proj.bias',
                      f'transformer.h.{l}.mlp.c_proj.bias')
            if w2b is not None: weights.append(w2b.flatten())

    lnf = get('model.norm.weight', 'transformer.ln_f.weight', 'ln_f.weight')
    if lnf is not None: weights.append(lnf.flatten())
    if arch['norm'] == 0:
        lnfb = get('model.norm.bias', 'transformer.ln_f.bias', 'ln_f.bias')
        if lnfb is not None: weights.append(lnfb.flatten())

    if not arch['weight_tie']:
        lm = get('lm_head.weight', 'transformer.lm_head.weight')
        if lm is not None: weights.append(lm.flatten())

    total = sum(w.size for w in weights)
    with open(output_path, 'wb') as f:
        f.write(np.array(header, dtype=np.int32).tobytes())
        for w in weights:
            f.write(w.astype(np.float32).tobytes())

    print(f"Converted: {output_path}")
    print(f"  arch: dim={D} hidden={H} layers={L} heads={NH} vocab={V}")
    print(f"  norm={'RMSNorm' if arch['norm'] else 'LayerNorm'}")
    print(f"  act={'SwiGLU' if arch['activation'] else 'GELU'}")
    print(f"  pos={'RoPE' if arch['pos_enc'] else 'Learned'}")
    print(f"  Total params: {total:,}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hf_model_dir> <output.bin>")
        sys.exit(1)
    convert_hf_to_baremetal(sys.argv[1], sys.argv[2])
