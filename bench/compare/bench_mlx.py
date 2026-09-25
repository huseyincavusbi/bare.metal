#!/usr/bin/env python3
"""Apple MLX inference benchmark -> baremetal.bench/v2 JSON.

Greedy, fixed prompt-tiling protocol (same as bench/bench.c). Requires mlx-lm
(`pip install mlx-lm`). Targets the common mlx_lm.stream_generate API.
"""
import argparse, json, time, platform
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

PROMPT_TEXT = (
    "Once upon a time, in a small village at the edge of a great forest, "
    "there lived a curious child who loved to ask questions about the world. "
    "Every morning the child would walk to the river and watch the water flow, "
    "wondering where it came from and where it was going. The villagers said "
    "the river came from the mountains, and the mountains came from the sky, "
    "and the sky held all the answers to every question ever asked."
)


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {"n": 0}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    def pc(p): return xs[min(n - 1, int(p / 100.0 * (n - 1) + 0.5))]
    return {"n": n, "min": xs[0], "max": xs[-1], "mean": mean,
            "stddev": var ** 0.5, "p50": pc(50), "p90": pc(90), "p99": pc(99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, tok = load(a.model)
    prompt_ids = tok.encode(PROMPT_TEXT)
    n_prompt = len(prompt_ids)
    sampler = make_sampler(temp=0.0)

    def run_once():
        ids, itls = [], []
        prev = first = None
        t0 = time.perf_counter()
        for resp in stream_generate(model, tok, prompt_ids, max_tokens=a.gen, sampler=sampler):
            now = time.perf_counter()
            if first is None:
                first = now
            ids.append(int(resp.token))
            if prev is not None:
                itls.append((now - prev) * 1000.0)
            prev = now
        return ids, itls, (first - t0) * 1000.0 if first else 0.0

    # warmup
    for _ in range(a.warmup):
        run_once()

    prefill_tps, decode_tps, ttft_ms, itl_all = [], [], [], []
    last_ids = None
    for _ in range(a.reps):
        ids, itls, ttft = run_once()
        dec = sum(itls) / 1000.0
        ttft_ms.append(ttft)
        itl_all += itls
        prefill_tps.append(n_prompt / (ttft / 1000.0) if ttft > 0 else 0.0)
        decode_tps.append((a.gen - 1) / dec if dec > 0 else 0.0)
        last_ids = ids

    text = tok.decode(last_ids)
    doc = {
        "schema": "baremetal.bench/v2",
        "meta": {"engine": "mlx", "model": a.model, "precision": "bf16",
                 "quant": "none", "host": platform.machine()},
        "config": {"prompt_tokens": n_prompt, "gen_tokens": a.gen,
                   "warmup": a.warmup, "reps": a.reps, "temp": 0.0},
        "results": {
            "prefill": {"tok_s": stats(prefill_tps)},
            "decode": {"tok_s": stats(decode_tps)},
            "ttft_ms": stats(ttft_ms),
            "itl_ms": stats(itl_all),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": last_ids,
            "generated_text": text,
        },
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
