import torch, numpy as np, struct, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/gemma3-270m"
prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
output = sys.argv[3] if len(sys.argv) > 3 else "test/gemma_ref.bin"

print(f"Loading {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to("cpu")
model.eval()

tokens = tokenizer.encode(prompt, return_tensors="pt")
inp, tgt = tokens[:, :-1], tokens[:, 1:]
print(f"Input: {inp[0].tolist()} Target: {tgt[0].tolist()} B={inp.shape[0]} T={inp.shape[1]}")

with torch.no_grad():
    logits = model(inp).logits

loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
print(f"Loss: {loss.item():.6f}")

B, T, V = logits.shape
logits_np = logits[0].numpy().astype(np.float32)
print(f"First 5 logits: {logits_np[0,:5]}")

os.makedirs(os.path.dirname(output), exist_ok=True)
with open(output, "wb") as f:
    f.write(np.array([B,T,V], dtype=np.int32).tobytes())
    f.write(inp[0].numpy().astype(np.int32).tobytes())
    f.write(tgt[0].numpy().astype(np.int32).tobytes())
    f.write(logits_np.tobytes())
print(f"Saved to {output} ({os.path.getsize(output)} bytes)")
