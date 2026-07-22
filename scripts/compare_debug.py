#!/usr/bin/env python3
"""Compare C debug dumps with HuggingFace reference intermediate states."""
import torch
import numpy as np
import struct
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def load_bin(path):
    """Load binary float32 tensor."""
    with open(path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32)

def compare_tensors(name, our, ref, tol=1e-4):
    """Compare two tensors and print diagnostics."""
    if our.shape != ref.shape:
        print(f"❌ {name}: shape mismatch {our.shape} vs {ref.shape}")
        return False
    
    diff = np.abs(our - ref)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / (np.max(np.abs(ref)) + 1e-8)
    
    match = max_diff < tol
    status = "✓" if match else "❌"
    print(f"{status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_diff={rel_diff:.6f}")
    
    if not match:
        print(f"   Our:   {our[:5]}")
        print(f"   Ref:   {ref[:5]}")
    
    return match

def main():
    model_dir = "data/smollm2-135m"
    prompt = "The quick brown fox"
    
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to("cpu")
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Prompt tokens: {tokens[0].tolist()}")
    
    # Get embedding
    with torch.no_grad():
        emb = model.model.embed_tokens(tokens).numpy()
        print(f"\nHF embedding shape: {emb.shape}")
        print(f"HF embedding[0,:5]: {emb[0,0,:5]}")
    
    # Run forward pass with hooks to capture intermediate states
    activations = {}
    
    def hook_input_layernorm(module, input, output):
        activations['l0_norm'] = output[0].detach().numpy()
    
    def hook_self_attn(module, input, output):
        activations['l0_attn'] = output[0].detach().numpy()
    
    def hook_post_attn(module, input, output):
        activations['l0_ffn'] = output[0].detach().numpy()
    
    # Register hooks
    model.model.layers[0].input_layernorm.register_forward_hook(hook_input_layernorm)
    model.model.layers[0].self_attn.register_forward_hook(hook_self_attn)
    model.model.layers[0].post_attention_layernorm.register_forward_hook(hook_post_attn)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=False)
    
    print(f"\n=== Comparing intermediate states ===\n")
    
    # Load our dumps
    our_emb = load_bin("debug/emb.bin")
    our_norm_weight = load_bin("debug/l0_norm_weight.bin")
    our_norm = load_bin("debug/l0_norm.bin")
    our_attn = load_bin("debug/l0_attn.bin")
    our_ffn = load_bin("debug/l0_ffn.bin")
    
    # Compare embedding (first token)
    hf_emb = emb[0, 0, :]  # batch=0, seq=0
    compare_tensors("embedding", our_emb, hf_emb)
    
    # Compare RMSNorm weight
    hf_norm_weight = model.model.layers[0].input_layernorm.weight.detach().numpy()
    compare_tensors("l0_norm_weight", our_norm_weight, hf_norm_weight)
    
    # Compare layer 0 norm output (first token, not last)
    if 'l0_norm' in activations:
        hf_norm = activations['l0_norm'][0, 0, :] if activations['l0_norm'].ndim == 3 else activations['l0_norm'][0, :]
        compare_tensors("l0_norm", our_norm, hf_norm)
    
    # Compare layer 0 attention output (first token)
    if 'l0_attn' in activations:
        hf_attn = activations['l0_attn'][0, 0, :] if activations['l0_attn'].ndim == 3 else activations['l0_attn'][0, :]
        compare_tensors("l0_attn", our_attn, hf_attn)
    
    # Compare layer 0 FFN output (first token)
    if 'l0_ffn' in activations:
        hf_ffn = activations['l0_ffn'][0, 0, :] if activations['l0_ffn'].ndim == 3 else activations['l0_ffn'][0, :]
        compare_tensors("l0_ffn", our_ffn, hf_ffn)
    
    print(f"\n=== Summary ===")
    print(f"Debug files: {list(Path('debug').glob('*.bin'))}")

if __name__ == "__main__":
    main()
