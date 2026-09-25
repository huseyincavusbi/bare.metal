#!/usr/bin/env python3
"""Collect baremetal.bench/v2 JSON files from a cross-engine run.

Prints a performance table, an output-consistency diff, and (with --out)
writes a single unified baremetal.compare/v1 JSON describing the whole run:
model, timestamp, settings, host, and every engine's stats + correctness.

Usage: collect.py [--out FILE] [--md FILE] [dir-or-json ...]
"""
import argparse, datetime, glob, json, os, sys

SCHEMA_RUN = "baremetal.bench/v2"
SCHEMA_CMP = "baremetal.compare/v1"


def load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            files.append(p)
    out = []
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if j.get("schema") == SCHEMA_RUN:
            j["_file"] = os.path.basename(f)
            out.append(j)
    return out


def g(j, *keys, default=0.0):
    cur = j
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def prefill(j):
    return g(j, "results", "prefill", "tok_s", "p50",
             default=g(j, "results", "prefill", "tok_s", "median"))


def decode(j):
    return g(j, "results", "decode", "tok_s", "p50",
             default=g(j, "results", "decode", "tok_s", "median"))


def diff_vs(ref, j):
    """Token-id or text-prefix agreement of j against ref."""
    rp, p = ref["results"].get("prompt_token_ids"), j["results"].get("prompt_token_ids")
    out = {}
    if rp and p:
        n = min(len(rp), len(p))
        first = next((i for i in range(n) if rp[i] != p[i]), None)
        out["prompt"] = "identical" if (first is None and len(rp) == len(p)) \
            else (f"first-div@{first}" if first is not None
                  else f"length {len(rp)}/{len(p)}")
    rid, id_ = ref["results"].get("generated_token_ids"), j["results"].get("generated_token_ids")
    rtx, tx = ref["results"].get("generated_text"), j["results"].get("generated_text")
    if rid and id_:
        n = min(len(rid), len(id_))
        first = next((i for i in range(n) if rid[i] != id_[i]), None)
        out["generated_tokens"] = {"n": n, "match_pct": 100.0 * (first if first is not None else n) / max(n, 1),
                                   "first_divergence": first}
    elif rtx and tx:
        n = min(len(rtx), len(tx))
        first = next((i for i in range(n) if rtx[i] != tx[i]), None)
        out["generated_chars"] = {"n": n, "prefix_pct": 100.0 * (first if first is not None else n) / max(n, 1),
                                  "first_divergence": first}
    return out


def engine_entry(ref, j):
    m, c, r = j["meta"], j["config"], j["results"]
    return {
        "engine": m.get("engine", j["_file"]),
        "file": j["_file"],
        "precision": m.get("precision"),
        "quant": m.get("quant"),
        "device": m.get("device"),
        "backend": {k: m[k] for k in ("torch", "mlx", "git_commit") if k in m},
        "config": c,
        "stats": {
            "prefill_tok_s": prefill(j),
            "decode_tok_s": decode(j),
            "ttft_ms": g(r, "ttft_ms", "p50", default=None),
            "itl_ms": g(r, "itl_ms", "p50", default=None),
            "load_ms": r.get("load_ms"),
            "rss_peak_bytes": g(r, "memory", "rss_peak_bytes", default=None),
            "model_bytes": g(r, "analysis", "model_bytes", default=None),
            "bytes_per_param": g(r, "analysis", "bytes_per_param", default=None),
        },
        "generated_text": r.get("generated_text"),
        "generated_token_ids": r.get("generated_token_ids"),
        "diff_vs_ref": diff_vs(ref, j),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["bench/compare/results"])
    ap.add_argument("--out", help="write unified comparison JSON here")
    ap.add_argument("--md", help="write markdown table here")
    a = ap.parse_args()

    runs = load(a.paths or ["bench/compare/results"])
    if not runs:
        print("(no baremetal.bench/v2 results found)")
        return

    # ---- table ----
    print(f"{'engine':<12}{'file':<24}{'prec':<6}{'q':<5}"
          f"{'pre t/s':>9}{'dec t/s':>9}{'ttft ms':>9}{'itl ms':>8}{'rss MB':>8}")
    for j in runs:
        m, r = j["meta"], j["results"]
        rss = g(r, "memory", "rss_peak_bytes", default=0) / 1e6
        print(f"{m.get('engine','?'):<12}{j['_file']:<24}{m.get('precision','?'):<6}"
              f"{m.get('quant','?'):<5}{prefill(j):>9.1f}{decode(j):>9.1f}"
              f"{g(r,'ttft_ms','p50',default=float('nan')):>9.2f}"
              f"{g(r,'itl_ms','p50',default=float('nan')):>8.2f}{rss:>8.0f}")

    # ---- diff ----
    ref = runs[0]
    print(f"\n--- output diff (greedy) vs {ref['meta'].get('engine', ref['_file'])} ---")
    for j in runs[1:]:
        d = diff_vs(ref, j)
        name = j["meta"].get("engine", j["_file"])
        line = f"  vs {name:<12}"
        if "prompt" in d:
            line += f" prompt:{d['prompt']}"
        gt = d.get("generated_tokens") or d.get("generated_chars")
        if gt:
            key = "token" if "generated_tokens" in d else "text"
            fd = gt["first_divergence"]
            pct = gt.get("match_pct", gt.get("prefix_pct"))
            line += f"  {key} match {pct:.1f}%  first-div {fd if fd is not None else '-'}"
        print(line)

    # ---- unified JSON ----
    if a.out:
        rm = ref["meta"]
        agg = {
            "schema": SCHEMA_CMP,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reference_engine": rm.get("engine", ref["_file"]),
            "settings": ref["config"],
            "host": {k: rm[k] for k in ("hw_model", "hw_machine", "gpu_name", "macos",
                                        "os_build", "ram_bytes", "p_cores", "e_cores")
                     if k in rm},
            "models": {j["meta"].get("engine", j["_file"]): j["meta"].get("model")
                       for j in runs},
            "engines": [engine_entry(ref, j) for j in runs],
        }
        json.dump(agg, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")

    if a.md:
        with open(a.md, "w") as f:
            f.write("| engine | precision | quant | prefill tok/s | decode tok/s | "
                    "ttft ms | itl ms | rss MB | token match |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for j in runs:
                e = engine_entry(ref, j)
                gt = e["diff_vs_ref"].get("generated_tokens")
                match = f"{gt['match_pct']:.1f}%" if gt else "-"
                rss = (e["stats"]["rss_peak_bytes"] or 0) / 1e6
                f.write(f"| {e['engine']} | {e['precision']} | {e['quant']} | "
                        f"{e['stats']['prefill_tok_s']:.1f} | {e['stats']['decode_tok_s']:.1f} | "
                        f"{(e['stats']['ttft_ms'] or float('nan')):.2f} | "
                        f"{(e['stats']['itl_ms'] or float('nan')):.2f} | {rss:.0f} | {match} |\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
