#!/usr/bin/env python3
"""Collect baremetal.bench/v2 JSON files and print a cross-engine comparison.

Usage: collect.py <dir-or-json> [...]
Prints a performance table and, when generated token ids / text are present,
an output diff against the first (reference) entry.
"""
import json, glob, os, sys


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
        if j.get("schema") == "baremetal.bench/v2":
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


def main():
    runs = load(sys.argv[1:] or ["bench/compare/results"])
    if not runs:
        print("(no baremetal.bench/v2 results found)")
        return
    print(f"{'engine':<12}{'file':<24}{'prec':<6}{'q':<5}"
          f"{'pre t/s':>9}{'dec t/s':>9}{'ttft ms':>9}{'itl ms':>8}{'rss MB':>8}")
    for j in runs:
        m, c, r = j["meta"], j["config"], j["results"]
        rss = g(r, "memory", "rss_peak_bytes", default=0) / 1e6
        print(f"{m.get('engine','?'):<12}{j['_file']:<24}{m.get('precision','?'):<6}"
              f"{m.get('quant','?'):<5}"
              f"{g(r,'prefill','tok_s','p50',default=g(r,'prefill','tok_s','median')):>9.1f}"
              f"{g(r,'decode','tok_s','p50',default=g(r,'decode','tok_s','median')):>9.1f}"
              f"{g(r,'ttft_ms','p50',default=float('nan')):>9.2f}"
              f"{g(r,'itl_ms','p50',default=float('nan')):>8.2f}"
              f"{rss:>8.0f}")

    # ---- output diff (greedy) vs the first entry ----
    ref = runs[0]
    ref_ids = ref["results"].get("generated_token_ids")
    ref_txt = ref["results"].get("generated_text")
    print("\n--- output diff (greedy) ---")
    for j in runs[1:]:
        ids = j["results"].get("generated_token_ids")
        txt = j["results"].get("generated_text")
        name = j["meta"].get("engine", j["_file"])
        if ref_ids and ids:
            n = min(len(ref_ids), len(ids))
            match = sum(1 for i in range(n) if ref_ids[i] == ids[i])
            first = next((i for i in range(n) if ref_ids[i] != ids[i]), None)
            print(f"  vs {name:<12} token-id match {100*match//max(n,1)}%  "
                  f"first-div {first if first is not None else '-'}  (n={n})")
        elif ref_txt and txt:
            same = ref_txt == txt
            print(f"  vs {name:<12} text {'IDENTICAL' if same else 'DIFFERS'} "
                  f"({len(ref_txt)} vs {len(txt)} chars)")
        else:
            print(f"  vs {name:<12} (no token ids/text recorded)")


if __name__ == "__main__":
    main()
