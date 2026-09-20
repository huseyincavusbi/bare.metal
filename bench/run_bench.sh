#!/usr/bin/env bash
# bench/run_bench.sh -- orchestrate bare.metal benchmarks.
#
# MUST run on real Apple Silicon hardware. The GitHub CI runners are paravirtual
# (VM GPU, no bf16), so their numbers are not representative.
#
# Each invocation writes into its own timestamped run directory under the
# results dir, so reruns never mix stale configurations into the summary.
#
# Energy is ON by default: it wraps runs in `sudo powermetrics` (needs
# passwordless sudo; skipped with a warning otherwise). The parser only counts
# samples inside the harness's measured active window, and two dedicated runs
# give per-phase energy (prefill-heavy and decode-heavy). Measured watts are
# propagated into every sweep file.
#
# Usage:
#   bench/run_bench.sh [options]
#
# Options:
#   --model DIR      model directory            (default: data/smollm2-135m)
#   --out DIR        base results directory     (default: bench/results)
#   --reps N         measured reps per config  (default: 5)
#   --warmup N       warmup reps                (default: 2)
#   --gen N          tokens to generate         (default: 128)
#   --quick          single config, fewer reps  (smoke)
#   --no-sweep       only the base config
#   --no-energy      skip powermetrics energy capture
set -euo pipefail

MODEL="${MODEL:-data/smollm2-135m}"
OUTDIR="${OUTDIR:-bench/results}"
REPS=5
WARMUP=2
GEN=128
SWEEP=1
QUICK=0
ENERGY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --model)     MODEL="$2"; shift 2;;
    --out)       OUTDIR="$2"; shift 2;;
    --reps)      REPS="$2"; shift 2;;
    --warmup)    WARMUP="$2"; shift 2;;
    --gen)       GEN="$2"; shift 2;;
    --no-sweep)  SWEEP=0; shift;;
    --no-energy) ENERGY=0; shift;;
    --quick)     QUICK=1; REPS=2; WARMUP=1; GEN=64; SWEEP=0; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_DIR="$OUTDIR/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
make bench >/dev/null

BENCH="./build/bench/bench"
[ -x "$BENCH" ] || { echo "bench binary missing" >&2; exit 1; }

host="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "host:    $host"
echo "model:   $MODEL"
echo "run dir: $RUN_DIR"
echo

# prompt_len:precision pairs to sweep
if [ "$SWEEP" -eq 1 ]; then
  CONFIGS="128:fp32 128:bf16 128:fp16 128:q8 512:bf16 1024:bf16 2048:bf16"
else
  CONFIGS="128:bf16"
fi
[ "$QUICK" -eq 1 ] && CONFIGS="64:bf16"

run_one() {
  local prompt="$1" prec="$2" tag="$3" extra="$4"
  local out="$RUN_DIR/${tag}.json"
  # shellcheck disable=SC2086
  $BENCH --model "$MODEL" --prompt-tokens "$prompt" --gen "$GEN" \
         --warmup "$WARMUP" --reps "$REPS" --seed 42 $extra --out "$out"
  echo "  wrote $out"
}

for cfg in $CONFIGS; do
  prompt="${cfg%%:*}"; prec="${cfg##*:}"
  extra=""
  if [ "$prec" = "q8" ]; then extra="--quant q8"; else extra="--precision $prec"; fi
  echo "== prompt=$prompt precision=$prec =="
  run_one "$prompt" "$prec" "p${prompt}_${prec}" "$extra"
done

# ---- energy capture (default on; powermetrics needs passwordless sudo) -----
HAVE_ENERGY=0
if [ "$ENERGY" -eq 1 ]; then
  if ! command -v powermetrics >/dev/null 2>&1; then
    echo "energy: powermetrics not found; skipping" >&2
  elif ! sudo -n true 2>/dev/null; then
    echo "energy: needs passwordless sudo for powermetrics; skipping" >&2
  else
    HAVE_ENERGY=1
  fi
fi

energy_run() {
  # $1 = output json; rest = bench args. Wraps the run in powermetrics and
  # merges avg power / residency / joules into the JSON. Always reaps the
  # privileged sampler, even if the benchmark fails.
  local out="$1"; shift
  local pmf; pmf="$(mktemp)"
  local pmlog="$RUN_DIR/$(basename "$out" .json).powermetrics.txt"
  sudo powermetrics --samplers cpu_power,gpu_power,thermal,ane_power --show-process-gpu \
       -i 250 -o "$pmf" >/dev/null 2>&1 &
  local pm_pid=$!
  sleep 1
  local rc=0
  # shellcheck disable=SC2086
  $BENCH --model "$MODEL" --seed 42 "$@" --out "$out" >/dev/null || rc=$?
  kill "$pm_pid" 2>/dev/null || true
  wait "$pm_pid" 2>/dev/null || true
  python3 - "$out" "$pmf" <<'PY' || true
import json, re, sys, datetime, os
out_path, pm_path = sys.argv[1], sys.argv[2]
d = json.load(open(out_path))
r, c = d["results"], d["config"]
t0 = r.get("active_started_unix", 0)
t1 = r.get("active_ended_unix", 0)

def parse_ts(s):
    try:
        return datetime.datetime.strptime(s.strip(), "%a %b %d %H:%M:%S %Y %z").timestamp()
    except Exception:
        return None

# Optional metrics: (json key, regex, unit scale). Absent fields are omitted, so
# the parser stays valid across macOS/powermetrics versions.
PATTERNS = [
    ("avg_cpu_w",       r'CPU Power:\s*([\d.]+)\s*mW', 0.001),
    ("avg_gpu_w",       r'GPU Power:\s*([\d.]+)\s*mW', 0.001),
    ("avg_ane_w",       r'ANE Power:\s*([\d.]+)\s*mW', 0.001),
    ("combined_w",      r'Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW', 0.001),
    ("gpu_active_pct",  r'GPU HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("gpu_freq_mhz",    r'GPU HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("p_cluster_w",     r'P\d*-Cluster Power:\s*([\d.]+)\s*mW', 0.001),
    ("e_cluster_w",     r'E\d*-Cluster Power:\s*([\d.]+)\s*mW', 0.001),
    ("p_residency_pct", r'P\d*-Cluster HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("e_residency_pct", r'E\d*-Cluster HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("p_freq_mhz",      r'P\d*-Cluster HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("e_freq_mhz",      r'E\d*-Cluster HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("cpu_die_c",       r'CPU die temperature:\s*([\d.]+)\s*C', 1.0),
    ("gpu_die_c",       r'GPU die temperature:\s*([\d.]+)\s*C', 1.0),
]

def new_block():
    return {"ts": None, "vals": {k: [] for (k, _, _) in PATTERNS}}

# Split the powermetrics stream into per-sample blocks.
blocks = []
cur = new_block()
for line in open(pm_path, errors='ignore'):
    m = re.search(r'\*\*\* Sampled system activity \(([^)]*)\)', line)
    if m:
        if cur["ts"] is not None:
            blocks.append(cur)
        cur = new_block(); cur["ts"] = parse_ts(m.group(1))
        continue
    if cur["ts"] is None:
        continue
    if "Average" in line or "(avg" in line:      # skip powermetrics' own averages
        continue
    for (key, rx, scale) in PATTERNS:
        mm = re.search(rx, line)
        if mm:
            cur["vals"][key].append(float(mm.group(1)) * scale)
if cur["ts"] is not None:
    blocks.append(cur)

# Only samples inside the harness's measured window (small tolerance);
# fall back to all samples if the window is unavailable.
sel = [b for b in blocks if t0 and t1 and (t0 - 0.5) <= b["ts"] <= (t1 + 0.5)]
if not sel:
    sel = blocks

def mean(key):
    xs = [v for b in sel for v in b["vals"][key]]
    return (sum(xs) / len(xs)) if xs else 0.0

energy = {"samples_used": len(sel)}
for (key, _, _) in PATTERNS:
    xs = [v for b in sel for v in b["vals"][key]]
    if xs:
        energy[key] = round(sum(xs) / len(xs), 2)

# Total power: measured SoC rails (CPU+GPU+ANE); fall back to the combined line.
cpu_w, gpu_w, ane_w = mean("avg_cpu_w"), mean("avg_gpu_w"), mean("avg_ane_w")
total_w = cpu_w + gpu_w + ane_w
if total_w == 0.0:
    total_w = mean("combined_w")
energy["avg_cpu_w"] = round(cpu_w, 2)
energy["avg_gpu_w"] = round(gpu_w, 2)
energy["avg_ane_w"] = round(ane_w, 2)
energy["avg_w"] = round(total_w, 2)
# raw powermetrics (incl. per-process GPU via --show-process-gpu) kept for
# inspection; per-process extraction is left to the reader for now.
energy["powermetrics_log"] = os.path.basename(out_path).replace(".json", ".powermetrics.txt")

active = r.get("active_seconds", 0.0)
reps = c.get("reps", 1)
joules = total_w * active
energy["joules"] = round(joules, 2)
if c.get("gen_tokens", 0) > 8:
    dec_tokens = (c["gen_tokens"] - 1) * reps
    energy["j_per_decode_token"] = round(joules / dec_tokens, 4) if dec_tokens else 0.0
if c.get("prompt_tokens", 0) > 128 and c.get("gen_tokens", 0) <= 8:
    pre_tokens = c["prompt_tokens"] * reps
    energy["j_per_prefill_token"] = round(joules / pre_tokens, 4) if pre_tokens else 0.0
gen = r.get("generated_tokens", 0)
energy["j_per_token"] = round(joules / gen, 4) if gen else 0.0
r["energy"] = energy
json.dump(d, open(out_path, "w"), indent=2)
extra = []
for k in ("cpu_die_c", "gpu_die_c", "gpu_freq_mhz", "gpu_active_pct"):
    if k in energy:
        extra.append(f"{k}={energy[k]}")
print(f"  {out_path}: {total_w:.2f} W (cpu {cpu_w:.2f} + gpu {gpu_w:.2f} + ane {ane_w:.2f}), "
      f"{joules:.1f} J, {len(sel)} samples" + (", " + ", ".join(extra) if extra else ""))
PY
  cp -f "$pmf" "$pmlog" 2>/dev/null || true
  rm -f "$pmf"
  return "$rc"
}

if [ "$HAVE_ENERGY" -eq 1 ]; then
  echo
  echo "== energy: decode-heavy run =="
  energy_run "$RUN_DIR/energy_decode.json" --prompt-tokens 128 --gen 512 \
             --warmup 1 --reps 1 --precision bf16
  echo "== energy: prefill-heavy run =="
  energy_run "$RUN_DIR/energy_prefill.json" --prompt-tokens 512 --gen 2 \
             --warmup 1 --reps 1 --precision bf16

  python3 - "$RUN_DIR" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
src = None
for cand in ("energy_decode.json", "energy_prefill.json"):
    p = os.path.join(d, cand)
    if os.path.exists(p):
        src = json.load(open(p)); break
if not src:
    raise SystemExit
avg_w = src["results"]["energy"]["avg_w"]
for f in glob.glob(os.path.join(d, "*.json")):
    if os.path.basename(f).startswith("energy_"):
        continue
    try: j = json.load(open(f))
    except Exception: continue
    if j.get("schema") != "baremetal.bench/v2": continue
    r = j["results"]
    active, gen = r.get("active_seconds", 0), r.get("generated_tokens", 0)
    if avg_w > 0 and active > 0:
        r["energy"] = {
            "avg_w": avg_w,
            "estimated": True,
            "joules": round(avg_w * active, 2),
            "j_per_token": round(avg_w * active / gen, 4) if gen else 0.0,
        }
        json.dump(j, open(f, "w"), indent=2)
print(f"  propagated {avg_w:.2f} W into all sweep results")
PY
fi

echo
echo "== summary =="
python3 - "$RUN_DIR" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: j = json.load(open(f))
    except Exception: continue
    if j.get("schema") != "baremetal.bench/v2": continue
    m, c, r = j["meta"], j["config"], j["results"]
    rows.append((os.path.basename(f), m["precision"], m["quant"],
                 c["prompt_tokens"], c["gen_tokens"],
                 r["prefill"]["tok_s"]["p50"], r["decode"]["tok_s"]["p50"],
                 r["ttft_ms"]["p50"], r["itl_ms"]["p50"],
                 r["memory"]["rss_peak_bytes"] / 1e6,
                 r.get("energy", {}).get("j_per_token", 0.0)))
if not rows:
    print("(no results)"); raise SystemExit
print(f"{'file':<24}{'prec':<6}{'q':<5}{'Np':>5}{'Ng':>5}{'pre t/s':>9}{'dec t/s':>9}"
      f"{'ttft ms':>9}{'itl ms':>8}{'rss MB':>8}{'J/tok':>8}")
for x in rows:
    print(f"{x[0]:<24}{x[1]:<6}{x[2]:<5}{x[3]:>5}{x[4]:>5}{x[5]:>9.1f}{x[6]:>9.1f}"
          f"{x[7]:>9.2f}{x[8]:>8.2f}{x[9]:>8.0f}{x[10]:>8.4f}")
PY
