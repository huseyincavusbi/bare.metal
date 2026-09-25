#!/usr/bin/env python3
"""llama.cpp inference benchmark -> baremetal.bench/v2 JSON.

Uses llama-bench for prefill/decode throughput (its `pp`/`tg` map exactly to our
prefill/decode) and llama-cli (greedy) for the generated text. Requires the
llama.cpp tools on PATH: llama-bench, llama-cli, and a GGUF model.
"""
import argparse, json, os, re, subprocess, sys

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


def bench_pp_tg(gguf, prompt_tokens, gen, reps):
    """Return (prefill_tps, decode_tps) from llama-bench json output."""
    r = run(["llama-bench", "-m", gguf, "-p", str(prompt_tokens),
             "-n", str(gen), "-r", str(reps), "-o", "json"])
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 0.0, 0.0
    try:
        data = json.loads(r.stdout)
    except Exception:
        return 0.0, 0.0
    pp = tg = 0.0
    for e in data:
        if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
            pp = float(e.get("avg_ts", 0.0))
        elif e.get("n_prompt", 0) == 0 and e.get("n_gen", 0) > 0:
            tg = float(e.get("avg_ts", 0.0))
    return pp, tg


def gen_text(gguf, prompt_tokens, gen, seed):
    """Greedy generation via llama-cli; returns (text, peak_rss_bytes)."""
    args = ["llama-completion", "-m", gguf, "-p", PROMPT_TEXT, "-n", str(gen),
            "--temp", "0", "--seed", str(seed), "--no-display-prompt",
            "-c", str(prompt_tokens + gen + 64)]
    r = run(["/usr/bin/time", "-l"] + args) if os.path.exists("/usr/bin/time") else run(args)
    text = r.stdout
    rss = 0
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    if m:
        rss = int(m.group(1))
    return text.strip(), rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="F16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pp, tg = bench_pp_tg(a.gguf, a.prompt_tokens, a.gen, a.reps)
    text, rss = gen_text(a.gguf, a.prompt_tokens, a.gen, a.seed)

    doc = {
        "schema": "baremetal.bench/v2",
        "meta": {"engine": "llama.cpp", "model": os.path.basename(a.gguf),
                 "precision": a.precision,
                 "quant": "none" if a.precision in ("F32", "F16", "BF16") else a.precision},
        "config": {"prompt_tokens": a.prompt_tokens, "gen_tokens": a.gen,
                   "reps": a.reps, "temp": 0.0, "seed": a.seed},
        "results": {
            "prefill": {"tok_s": {"p50": pp}},
            "decode": {"tok_s": {"p50": tg}},
            "memory": {"rss_peak_bytes": rss},
            "generated_text": text,
        },
    }
    json.dump(doc, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}  (pp={pp:.1f} tg={tg:.1f} t/s)")


if __name__ == "__main__":
    main()
