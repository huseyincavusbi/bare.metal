#!/usr/bin/env python3
"""llama.cpp inference benchmark -> baremetal.bench/v2 JSON.

Uses llama-bench for prefill/decode throughput (its `pp`/`tg` map exactly to our
prefill/decode) and llama-completion (greedy) for the generated text. Requires
the llama.cpp tools on PATH: llama-bench, llama-tokenize, llama-completion, and
a GGUF model.
"""
import argparse, json, os, re, subprocess, sys, time, resource

PROMPT_TEXT = (
    "Once upon a time, in a small village at the edge of a great forest, "
    "there lived a curious child who loved to ask questions about the world. "
    "Every morning the child would walk to the river and watch the water flow, "
    "wondering where it came from and where it was going. The villagers said "
    "the river came from the mountains, and the mountains came from the sky, "
    "and the sky held all the answers to every question ever asked."
)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def tokenize(gguf, text):
    """Token ids for `text` via llama-tokenize (no BOS for these models)."""
    r = run(["llama-tokenize", "-m", gguf, "-p", text, "--ids"])
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return []


def bench_pp_tg(gguf, prompt_tokens, gen, reps):
    """Return (prefill_tps, decode_tps, model_size, n_params) from llama-bench."""
    r = run(["llama-bench", "-m", gguf, "-p", str(prompt_tokens),
             "-n", str(gen), "-r", str(reps), "-o", "json"])
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 0.0, 0.0, 0, 0
    try:
        data = json.loads(r.stdout)
    except Exception:
        return 0.0, 0.0, 0, 0
    pp = tg = 0.0
    model_size = n_params = 0
    for e in data:
        model_size = model_size or int(e.get("model_size", 0))
        n_params = n_params or int(e.get("model_n_params", 0))
        if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
            pp = float(e.get("avg_ts", 0.0))
        elif e.get("n_prompt", 0) == 0 and e.get("n_gen", 0) > 0:
            tg = float(e.get("avg_ts", 0.0))
    return pp, tg, model_size, n_params


def gen_text(gguf, prompt_tokens, gen, seed):
    """Greedy generation via llama-completion; returns (text, peak_rss_bytes)."""
    args = ["llama-completion", "-m", gguf, "-p", PROMPT_TEXT, "-n", str(gen),
            "--temp", "0", "--seed", str(seed), "--no-display-prompt",
            "-c", str(prompt_tokens + gen + 64)]
    r = run(["/usr/bin/time", "-l"] + args) if os.path.exists("/usr/bin/time") else run(args)
    text = r.stdout
    rss = 0
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    if m:
        rss = int(m.group(1))
    return text.rstrip(), rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="F16")
    ap.add_argument("--peak-gflops", type=float, default=4200.0)
    ap.add_argument("--peak-gbps", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompt_ids = tokenize(a.gguf, PROMPT_TEXT)
    n_prompt = len(prompt_ids)

    active_started = time.time()
    pp, tg, model_size, n_params = bench_pp_tg(a.gguf, n_prompt, a.gen, a.reps)
    tf = time.perf_counter()
    text, rss = gen_text(a.gguf, n_prompt, a.gen, a.seed)
    first_call_ms = (time.perf_counter() - tf) * 1000.0
    active_ended = time.time()
    model_size = model_size or os.path.getsize(a.gguf)
    gen_ids = tokenize(a.gguf, text) if text else []

    # llama-bench reports pp/tg rates; derive TTFT and ITL from them.
    ttft_ms = n_prompt / pp * 1000.0 if pp > 0 else 0.0
    itl_ms = 1000.0 / tg if tg > 0 else 0.0

    # decode bytes/token ≈ whole model read per token
    dec_bytes_s = model_size * tg
    achieved_gbps = dec_bytes_s / 1e9 if tg > 0 else 0.0
    gflops = 2.0 * n_params * tg / 1e9 if (n_params and tg > 0) else 0.0

    doc = {
        "schema": "baremetal.bench/v2",
        "meta": {"engine": "llama.cpp", "model": os.path.basename(a.gguf),
                 "precision": a.precision,
                 "quant": "none" if a.precision in ("F32", "F16", "BF16") else a.precision},
        "config": {"prompt_tokens": n_prompt, "gen_tokens": a.gen,
                   "reps": a.reps, "temp": 0.0, "seed": a.seed},
        "results": {
            "first_call_ms": round(first_call_ms, 3),
            "prefill": {"tok_s": {"p50": pp}},
            "decode": {"tok_s": {"p50": tg}},
            "ttft_ms": {"p50": round(ttft_ms, 3)},
            "itl_ms": {"p50": round(itl_ms, 3)},
            "memory": {"rss_peak_bytes": rss or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            "analysis": {
                "model_bytes": model_size,
                "bytes_per_param": round(model_size / n_params, 3) if n_params else 0.0,
                "achieved_gbps": round(achieved_gbps, 3),
                "gflops": round(gflops, 2),
                "arith_intensity": round((2.0 * n_params) / model_size, 3) if model_size else 0.0,
                "mfu_pct_of_peak": round(100.0 * gflops / a.peak_gflops, 2) if a.peak_gflops else 0.0,
                "bw_pct_of_peak": round(100.0 * achieved_gbps / a.peak_gbps, 2) if a.peak_gbps else 0.0,
            },
            "active_started_unix": round(active_started, 3),
            "active_ended_unix": round(active_ended, 3),
            "active_seconds": round(active_ended - active_started, 3),
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": gen_ids,
            "generated_text": text,
            "generated_tokens": len(gen_ids),
        },
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}  (pp={pp:.1f} tg={tg:.1f} t/s, prompt={n_prompt})")


if __name__ == "__main__":
    main()
