import torch
import numpy as np
import struct, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_debug_state(model_dir, prompt, output_path):
    print(f"Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32
    ).to("cpu")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    print(f"Input: {input_ids.shape}, Targets: {targets.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Target tokens: {targets[0].tolist()}")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

    print(f"Loss: {loss.item():.6f}")
    print(f"Logits shape: {logits.shape}")

    logits_np = logits[0].numpy().astype(np.float32)
    input_np = input_ids[0].numpy().astype(np.int32)
    target_np = targets[0].numpy().astype(np.int32)

    with open(output_path, "wb") as f:
        B, T, V = logits.shape[0], logits.shape[1], logits.shape[2]
        header = np.array([B, T, V], dtype=np.int32)
        f.write(header.tobytes())
        f.write(input_np.tobytes())
        f.write(target_np.tobytes())
        f.write(logits_np.tobytes())

    print(f"Saved debug state to {output_path} ({os.path.getsize(output_path)} bytes)")

    # Also print first few logits for manual comparison
    print(f"First 5 logits of first token: {logits_np[0,:5]}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model_dir> <prompt> <output.bin>")
        sys.exit(1)
    save_debug_state(sys.argv[1], sys.argv[2], sys.argv[3])
