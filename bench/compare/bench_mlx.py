#!/usr/bin/env python3
"""Apple MLX inference benchmark -> baremetal.bench/v2 JSON.

Greedy, raw prompt (no tiling). Requires mlx-lm. Targets the common
mlx_lm.stream_generate API.
"""
import argparse, json, time, platform, resource
import mlx.core as mx
from mlx.utils import tree_flatten
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
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--peak-gflops", type=float, default=4200.0)
    ap.add_argument("--peak-gbps", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load(a.model)
    load_ms = (time.perf_counter() - t0) * 1000.0

    weights = tree_flatten(model.parameters())
    params = sum(v.size for _, v in weights)
    model_bytes = sum(v.nbytes for _, v in weights)
    mx.reset_peak_memory()

    prompt_ids = tok.encode(PROMPT_TEXT)
    n_prompt = len(prompt_ids)
    sampler = make_sampler(temp=0.0)

    def run_once():
        ids, itls = [], []
        prev = first = None
        t = time.perf_counter()
        for resp in stream_generate(model, tok, prompt_ids, max_tokens=a.gen, sampler=sampler):
            now = time.perf_counter()
            if first is None:
                first = now
            ids.append(int(resp.token))
            if prev is not None:
                itls.append((now - prev) * 1000.0)
            prev = now
        return ids, itls, (first - t) * 1000.0 if first else 0.0

    first_call_ms = None
    for _ in range(a.warmup):
        t = time.perf_counter()
        run_once()
        if first_call_ms is None:
            first_call_ms = (time.perf_counter() - t) * 1000.0

    prefill_tps, decode_tps, ttft_ms, itl_all = [], [], [], []
    last_ids = None
    active_started = time.time()
    active_s = 0.0
    for _ in range(a.reps):
        rt = time.perf_counter()
        ids, itls, ttft = run_once()
        active_s += time.perf_counter() - rt
        dec = sum(itls) / 1000.0
        ttft_ms.append(ttft)
        itl_all += itls
        prefill_tps.append(n_prompt / (ttft / 1000.0) if ttft > 0 else 0.0)
        decode_tps.append((a.gen - 1) / dec if dec > 0 else 0.0)
        last_ids = ids
    active_ended = time.time()

    text = tok.decode(last_ids)
    itl_mean = sum(itl_all) / len(itl_all) if itl_all else 0.0
    itl_s = itl_mean / 1000.0
    achieved_gbps = model_bytes / itl_s / 1e9 if itl_s > 0 else 0.0
    gflops = 2.0 * params / itl_s / 1e9 if itl_s > 0 else 0.0
    doc = {
        "schema": "baremetal.bench/v2",
        "meta": {"engine": "mlx", "model": a.model, "precision": "bf16",
                 "quant": "none", "host": platform.machine(),
                 "mlx": getattr(mx, "__version__", None)},
        "config": {"prompt_tokens": n_prompt, "gen_tokens": a.gen,
                   "warmup": a.warmup, "reps": a.reps, "temp": 0.0},
        "results": {
            "load_ms": round(load_ms, 3),
            "first_call_ms": round(first_call_ms, 3) if first_call_ms else None,
            "prefill": {"tok_s": stats(prefill_tps)},
            "decode": {"tok_s": stats(decode_tps)},
            "ttft_ms": stats(ttft_ms),
            "itl_ms": stats(itl_all),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": last_ids,
            "generated_text": text,
            "memory": {"rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "gpu_peak_bytes": mx.get_peak_memory() or None},
            "analysis": {
                "model_bytes": model_bytes,
                "bytes_per_param": round(model_bytes / params, 3) if params else 0.0,
                "achieved_gbps": round(achieved_gbps, 3),
                "gflops": round(gflops, 2),
                "arith_intensity": round((2.0 * params) / model_bytes, 3) if model_bytes else 0.0,
                "mfu_pct_of_peak": round(100.0 * gflops / a.peak_gflops, 2) if a.peak_gflops else 0.0,
                "bw_pct_of_peak": round(100.0 * achieved_gbps / a.peak_gbps, 2) if a.peak_gbps else 0.0,
            },
            "active_started_unix": round(active_started, 3),
            "active_ended_unix": round(active_ended, 3),
            "active_seconds": round(active_s, 3),
            "generated_tokens": len(last_ids),
        },
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
