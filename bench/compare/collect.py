#!/usr/bin/env python3
"""Collect baremetal.bench/v2 results.

Two modes:

1. Directory mode (legacy): print a table + output diff for a folder of v2 JSONs
   and optionally write a single comparison file.
       collect.py [--out FILE | --outdir DIR] [--benchmark LABEL] [dir ...]

2. Run-dir mode: aggregate a bench/benchmarks/<...> run folder into
       run.json, grand.json, models/<stem>.json, engines/<engine>.json
   and print one table + diff per model.
       collect.py --run-dir DIR [--settings STR]
"""
import argparse, datetime, glob, json, os, re, subprocess, sys

SCHEMA_RUN = "baremetal.bench/v2"
SCHEMA_CMP = "baremetal.compare/v1"
SCHEMA_ALL = "baremetal.benchmark/v1"
SCHEMA_MODEL = "baremetal.model/v1"
SCHEMA_ENGINE = "baremetal.engine/v1"


# ---------------------------------------------------------------- helpers ----
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


def san(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def diff_vs(ref, j):
    rp, p = ref["results"].get("prompt_token_ids"), j["results"].get("prompt_token_ids")
    out = {}
    if rp and p:
        n = min(len(rp), len(p))
        first = next((i for i in range(n) if rp[i] != p[i]), None)
        out["prompt"] = "identical" if (first is None and len(rp) == len(p)) \
            else (f"first-div@{first}" if first is not None else f"length {len(rp)}/{len(p)}")
    rid, id_ = ref["results"].get("generated_token_ids"), j["results"].get("generated_token_ids")
    rtx, tx = ref["results"].get("generated_text"), j["results"].get("generated_text")
    if rid and id_:
        n = min(len(rid), len(id_))
        first = next((i for i in range(n) if rid[i] != id_[i]), None)
        out["generated_tokens"] = {"n": n,
                                   "match_pct": 100.0 * (first if first is not None else n) / max(n, 1),
                                   "first_divergence": first}
    elif rtx and tx:
        n = min(len(rtx), len(tx))
        first = next((i for i in range(n) if rtx[i] != tx[i]), None)
        out["generated_chars"] = {"n": n,
                                  "prefix_pct": 100.0 * (first if first is not None else n) / max(n, 1),
                                  "first_divergence": first}
    return out


def summary(j):
    r = j["results"]
    return {
        "prefill_tok_s": prefill(j),
        "decode_tok_s": decode(j),
        "ttft_ms": g(r, "ttft_ms", "p50", default=None),
        "itl_ms": g(r, "itl_ms", "p50", default=None),
        "load_ms": r.get("load_ms"),
        "first_call_ms": r.get("first_call_ms"),
        "rss_peak_bytes": g(r, "memory", "rss_peak_bytes", default=None),
        "gpu_peak_bytes": g(r, "memory", "gpu_peak_bytes", default=None),
        "model_bytes": g(r, "analysis", "model_bytes", default=None),
        "bytes_per_param": g(r, "analysis", "bytes_per_param", default=None),
        "achieved_gbps": g(r, "analysis", "achieved_gbps", default=None),
        "mfu_pct_of_peak": g(r, "analysis", "mfu_pct_of_peak", default=None),
        "gpu_busy_pct": g(r, "gpu_exec", "busy_pct", default=None),
        "combined_w": g(r, "energy", "combined_w", default=None),
    }


def engine_entry(ref, j):
    m = j["meta"]
    raw = {k: j[k] for k in ("meta", "config", "results") if k in j}
    return {
        "engine": m.get("engine", j["_file"]),
        "file": j["_file"],
        "model": m.get("model"),
        "precision": m.get("precision"),
        "quant": m.get("quant"),
        "diff_vs_ref": diff_vs(ref, j) if ref else {},
        "summary": summary(j),
        "raw": raw,
    }


def print_table(runs):
    print(f"{'engine':<12}{'file':<26}{'prec':<7}{'q':<6}"
          f"{'pre t/s':>9}{'dec t/s':>9}{'ttft ms':>9}{'itl ms':>8}{'rss MB':>8}{'W':>7}")
    for j in runs:
        m, r = j["meta"], j["results"]
        print(f"{m.get('engine','?'):<12}{j['_file']:<26}{str(m.get('precision','?')):<7}"
              f"{str(m.get('quant','?')):<6}{prefill(j):>9.1f}{decode(j):>9.1f}"
              f"{g(r,'ttft_ms','p50',default=float('nan')):>9.2f}"
              f"{g(r,'itl_ms','p50',default=float('nan')):>8.2f}"
              f"{g(r,'memory','rss_peak_bytes',default=0)/1e6:>8.0f}"
              f"{g(r,'energy','combined_w',default=float('nan')):>7.2f}")


def print_diff(ref, runs):
    print(f"  --- output diff vs {ref['meta'].get('engine', ref['_file'])} ---")
    for j in runs:
        if j is ref:
            continue
        d = diff_vs(ref, j)
        line = f"    vs {j.get('_file', j['meta'].get('engine', '?')):<24}"
        if "prompt" in d:
            line += f" prompt:{d['prompt']}"
        gt = d.get("generated_tokens") or d.get("generated_chars")
        if gt:
            key = "token" if "generated_tokens" in d else "text"
            line += (f"  {key} match {gt.get('match_pct', gt.get('prefix_pct')):.1f}%"
                     f"  first-div {gt['first_divergence'] if gt['first_divergence'] is not None else '-'}")
        print(line)


# --------------------------------------------------------- run-dir mode ------
def parse_raw(run_dir):
    models = {}
    for stem_dir in sorted(glob.glob(os.path.join(run_dir, "raw", "*"))):
        stem = os.path.basename(stem_dir)
        m = models.setdefault(stem, {"engines": {}, "sweep": {}})
        for f in sorted(glob.glob(os.path.join(stem_dir, "*.json"))):
            base = os.path.basename(f)[:-5]
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if j.get("schema") != SCHEMA_RUN:
                continue
            if base.startswith("sweep__"):
                m["sweep"][base[len("sweep__"):]] = j
            elif "__" in base:
                eng, fmt = base.split("__", 1)
                m["engines"].setdefault(eng, {})[fmt] = j
    return models


def git_probe():
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    return {"describe": run("git describe --tags --always --dirty 2>/dev/null"),
            "commit": run("git rev-parse --short HEAD 2>/dev/null"),
            "branch": run("git rev-parse --abbrev-ref HEAD 2>/dev/null")}


def first_host(models):
    for stem in models:
        for eng in models[stem]["engines"]:
            for fmt in models[stem]["engines"][eng]:
                m = models[stem]["engines"][eng][fmt]["meta"]
                return {k: m[k] for k in ("hw_model", "hw_machine", "gpu_name", "gpu_cores",
                                          "macos", "os_build", "ram_bytes", "p_cores", "e_cores")
                        if k in m}
    return {}


def aggregate(run_dir, settings_str):
    models = parse_raw(run_dir)
    if not models:
        print(f"(no raw results under {run_dir}/raw)")
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    run_meta = {
        "name": os.path.basename(run_dir),
        "generated_at": now.isoformat(),
        "version": git_probe(),
        "host": first_host(models),
        "settings": settings_str,
        "models": sorted(models.keys()),
    }

    # per-model tables + diff
    for stem in sorted(models):
        runs = []
        for eng in ("bare.metal", "llama.cpp", "mlx", "torch-mps"):
            for fmt, j in sorted(models[stem]["engines"].get(eng, {}).items()):
                jj = dict(j); jj["_file"] = f"{eng}__{fmt}"
                runs.append(jj)
        if not runs:
            continue
        print(f"\n=== {stem} ===")
        print_table(runs)
        ref = next((j for j in runs if j["meta"].get("engine") == "bare.metal"
                    and j["meta"].get("quant") in ("none", None)), runs[0])
        print_diff(ref, runs)
        if models[stem]["sweep"]:
            print(f"  sweep configs: {', '.join(sorted(models[stem]['sweep']))}")

    # grand
    grand_models = {}
    for stem, data in sorted(models.items()):
        engines = {}
        for eng, variants in sorted(data["engines"].items()):
            engines[eng] = {"variants": {fmt: {k: j[k] for k in ("meta", "config", "results") if k in j}
                                         for fmt, j in sorted(variants.items())}}
        sweep = {k: {kk: v[kk] for kk in ("meta", "config", "results") if kk in v}
                 for k, v in sorted(data["sweep"].items())}
        grand_models[stem] = {"engines": engines, "sweep": sweep}
    grand = {"schema": SCHEMA_ALL, "run": run_meta, "models": grand_models}
    json.dump(grand, open(os.path.join(run_dir, "grand.json"), "w"), indent=2)
    json.dump(run_meta, open(os.path.join(run_dir, "run.json"), "w"), indent=2)

    # per-model files
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    for stem, data in grand_models.items():
        json.dump({"schema": SCHEMA_MODEL, "run": run_meta, "model": stem, **data},
                  open(os.path.join(run_dir, "models", f"{stem}.json"), "w"), indent=2)

    # per-engine files
    os.makedirs(os.path.join(run_dir, "engines"), exist_ok=True)
    engines = {}
    for stem, data in grand_models.items():
        for eng, body in data["engines"].items():
            engines.setdefault(eng, {})[stem] = body
    for eng, body in engines.items():
        json.dump({"schema": SCHEMA_ENGINE, "run": run_meta, "engine": eng, "models": body},
                  open(os.path.join(run_dir, "engines", f"{san(eng)}.json"), "w"), indent=2)

    print(f"\nwrote {run_dir}/run.json, grand.json, models/*.json, engines/*.json")


# ------------------------------------------------------------- old mode ------
def build(ref, runs, benchmark, date_override=None):
    rm = ref["meta"]
    now = datetime.datetime.now(datetime.timezone.utc)
    date = date_override or datetime.datetime.now().strftime("%m-%d-%Y")
    engines = [j["meta"].get("engine", j["_file"]) for j in runs]
    model = san(os.path.basename(str(rm.get("model", "model"))))
    fname = f"{model}x{san('+'.join(engines))}x{san(date)}x{san(benchmark)}.json"
    return {
        "schema": SCHEMA_CMP,
        "run": {
            "name": fname[:-5],
            "generated_at": now.isoformat(),
            "benchmark": benchmark,
            "reference_engine": rm.get("engine", ref["_file"]),
            "engines": engines,
            "model": rm.get("model"),
        },
        "settings": ref["config"],
        "host": {k: rm[k] for k in ("hw_model", "hw_machine", "gpu_name", "gpu_cores",
                                    "macos", "os_build", "ram_bytes", "p_cores", "e_cores",
                                    "gpu_max_threads_per_tg", "gpu_max_threadgroup_mem")
                 if k in rm},
        "models": {j["meta"].get("engine", j["_file"]): j["meta"].get("model") for j in runs},
        "engines": [engine_entry(ref, j) for j in runs],
    }, fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["bench/compare/results"])
    ap.add_argument("--run-dir", help="aggregate a bench/benchmarks/<...> run folder")
    ap.add_argument("--settings", default="", help="settings string recorded in run.json")
    ap.add_argument("--outdir", help="dir to write <model>x<engines>x<date>x<benchmark>.json")
    ap.add_argument("--benchmark", default="cross-engine")
    ap.add_argument("--date", help="override date token in the filename")
    ap.add_argument("--out", help="explicit output path (overrides --outdir naming)")
    ap.add_argument("--md", help="also write a markdown table here")
    a = ap.parse_args()

    if a.run_dir:
        aggregate(a.run_dir, a.settings)
        return

    runs = load(a.paths or ["bench/compare/results"])
    if not runs:
        print("(no baremetal.bench/v2 results found)")
        return
    ref = runs[0]
    print_table(runs)
    print(f"\n--- output diff (greedy) vs {ref['meta'].get('engine', ref['_file'])} ---")
    print_diff(ref, runs)

    agg, fname = build(ref, runs, a.benchmark, a.date)
    path = a.out or (os.path.join(a.outdir, fname) if a.outdir else None)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(agg, open(path, "w"), indent=2)
        print(f"\nwrote {path}")

    if a.md:
        with open(a.md, "w") as f:
            f.write("| engine | precision | quant | prefill tok/s | decode tok/s | "
                    "ttft ms | itl ms | rss MB | token match |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for e in agg["engines"]:
                gt = e["diff_vs_ref"].get("generated_tokens")
                s = e["summary"]
                f.write(f"| {e['engine']} | {e['precision']} | {e['quant']} | "
                        f"{s['prefill_tok_s']:.1f} | {s['decode_tok_s']:.1f} | "
                        f"{(s['ttft_ms'] or float('nan')):.2f} | "
                        f"{(s['itl_ms'] or float('nan')):.2f} | "
                        f"{(s['rss_peak_bytes'] or 0)/1e6:.0f} | "
                        f"{(str(round(gt['match_pct'],1))+'%') if gt else '-'} |\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
