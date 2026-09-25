#!/usr/bin/env python3
"""PyTorch MPS inference benchmark -> baremetal.bench/v2 JSON.

Greedy, fixed seed, same prompt-tiling protocol as bench/bench.c so results are
comparable. Requires: torch, transformers.
"""
import argparse, json, time, platform
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


def build_prompt_ids(tok, n):
    ids = tok(PROMPT_TEXT)["input_ids"]
    return [ids[i % len(ids)] for i in range(n)]


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {"n": 0}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n

    def pc(p):
        return xs[min(n - 1, int(p / 100.0 * (n - 1) + 0.5))]
    return {"n": n, "min": xs[0], "max": xs[-1], "mean": mean,
            "stddev": var ** 0.5, "p50": pc(50), "p90": pc(90), "p99": pc(99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()

    ids = torch.tensor([build_prompt_ids(tok, a.prompt_tokens)], device=dev)

    def run_once():
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(ids)
            t1 = time.perf_counter()
            nxt = int(out.logits[0, -1].argmax())
            t2 = time.perf_counter()
            gen = [nxt]
            itls = []
            past = out.past_key_values
            cur = torch.tensor([[nxt]], device=dev)
            for _ in range(a.gen - 1):
                ta = time.perf_counter()
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                cur = out.logits[0, -1].argmax().view(1, 1)
                itls.append((time.perf_counter() - ta) * 1000.0)
                gen.append(int(cur))
        return (t1 - t0) * 1000.0, (t2 - t0) * 1000.0, itls, gen

    for _ in range(a.warmup):
        run_once()

    pre, ttft, itl_all, tps, last = [], [], [], [], None
    for _ in range(a.reps):
        p, tt, itls, gen = run_once()
        pre.append(p)
        ttft.append(tt)
        itl_all += itls
        d = sum(itls) / 1000.0
        tps.append((a.gen - 1) / d if d > 0 else 0.0)
        last = gen

    text = tok.decode(last, skip_special_tokens=True)
    doc = {
        "schema": "baremetal.bench/v2",
        "meta": {
            "engine": "torch-mps", "device": dev,
            "model": a.model, "precision": "fp16" if dev == "mps" else "fp32",
            "quant": "none", "host": platform.processor() or platform.machine(),
        },
        "config": {"prompt_tokens": a.prompt_tokens, "gen_tokens": a.gen,
                   "warmup": a.warmup, "reps": a.reps, "temp": 0.0, "seed": a.seed},
        "results": {
            "prefill": {"tok_s": stats(tps)},
            "decode": {"tok_s": stats(tps)},
            "ttft_ms": stats(ttft),
            "itl_ms": stats(itl_all),
            "generated_token_ids": last,
            "generated_text": text,
        },
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
