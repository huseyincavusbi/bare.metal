#!/usr/bin/env python3
"""Dump a bare.metal architecture parity fixture from a HuggingFace model.

Produces:
  <out-prefix>.json        prompt ids, greedy token ids, fixture metadata
  <out-prefix>.logits.f32  full-vocabulary logits (float32, little-endian)

The prompt and tiling match bench/bench.c so a fixture doubles as the
reference for the benchmark prompt.
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEXT = (
    "Once upon a time, in a small village at the edge of a great forest, "
    "there lived a curious child who loved to ask questions about the world. "
    "Every morning the child would walk to the river and watch the water flow, "
    "wondering where it came from and where it was going. The villagers said "
    "the river came from the mountains, and the mountains came from the sky, "
    "and the sky held all the answers to every question ever asked."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model directory")
    ap.add_argument("--out-prefix", required=True, help="e.g. test/parity/smollm2-135m")
    ap.add_argument("--prompt-tokens", type=int, default=88)
    ap.add_argument("--raw-prompt", action="store_true")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(PROMPT_TEXT)["input_ids"]
    if not ids:
        raise SystemExit("tokenizer produced no ids")

    if args.raw_prompt:
        prompt = ids[: args.prompt_tokens]
    else:
        prompt = [ids[i % len(ids)] for i in range(args.prompt_tokens)]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt], device=args.device), use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float().cpu().numpy()
        nxt = int(np.argmax(logits))
        generated = [nxt]
        for _ in range(args.gen - 1):
            out = model(
                input_ids=torch.tensor([[nxt]], device=args.device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            lg = out.logits[0, -1].float().cpu().numpy()
            nxt = int(np.argmax(lg))
            generated.append(nxt)

    name = os.path.basename(args.model.rstrip("/"))
    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    logits_name = name + ".logits.f32"

    logits.astype("<f4").tofile(args.out_prefix + ".logits.f32")

    fixture = {
        "format": "baremetal.parity/v1",
        "model": name,
        "precision": "fp32",
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "logits_file": logits_name,
        "vocab_size": int(logits.shape[0]),
    }
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(
        f"wrote {args.out_prefix}.json ({len(prompt)} prompt, {len(generated)} gen) "
        f"and {args.out_prefix}.logits.f32 ({logits.shape[0]} floats)"
    )


if __name__ == "__main__":
    main()
