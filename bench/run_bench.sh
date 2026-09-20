#!/usr/bin/env bash
# bench/run_bench.sh -- orchestrate bare.metal benchmarks.
#
# MUST run on real Apple Silicon hardware. The GitHub CI runners are paravirtual
# (VM GPU, no bf16), so their numbers are not representative.
#
# Usage:
#   bench/run_bench.sh [options]
#
# Options:
#   --model DIR      model directory            (default: data/smollm2-135m)
#   --out DIR        results directory          (default: bench/results)
#   --reps N         measured reps per config   (default: 5)
#   --warmup N       warmup reps                (default: 2)
#   --gen N          tokens to generate         (default: 128)
#   --quick          single config, fewer reps  (smoke)
#   --energy         also measure energy via powermetrics (needs sudo)
#   --no-sweep       only run the base config
#
# Each configuration produces one JSON file (baremetal.bench/v1 schema) under
# the results directory.
set -euo pipefail

MODEL="${MODEL:-data/smollm2-135m}"
OUTDIR="${OUTDIR:-bench/results}"
REPS=5
WARMUP=2
GEN=128
ENERGY=0
SWEEP=1
QUICK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model)  MODEL="$2"; shift 2;;
    --out)    OUTDIR="$2"; shift 2;;
    --reps)   REPS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --gen)    GEN="$2"; shift 2;;
    --energy) ENERGY=1; shift;;
    --no-sweep) SWEEP=0; shift;;
    --quick)  QUICK=1; REPS=2; WARMUP=1; GEN=64; SWEEP=0; shift;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p "$OUTDIR"
make bench >/dev/null

BENCH="./build/bench/bench"
[ -x "$BENCH" ] || { echo "bench binary missing" >&2; exit 1; }

host="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "host:   $host"
echo "model:  $MODEL"
echo "outdir: $OUTDIR"
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
  local out="$OUTDIR/${tag}.json"
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

# ---- optional energy measurement (Apple-specific, needs sudo) --------------
if [ "$ENERGY" -eq 1 ]; then
  if ! command -v powermetrics >/dev/null 2>&1; then
    echo "powermetrics not found; skipping energy" >&2
  elif ! sudo -n true 2>/dev/null; then
    echo "energy: needs passwordless sudo for powermetrics; skipping" >&2
  else
    pmf="$(mktemp)"
    echo "== energy run (powermetrics) =="
    sudo powermetrics --samplers gpu_power,cpu_power -i 500 -o "$pmf" >/dev/null 2>&1 &
    pm_pid=$!
    sleep 2
    eout="$OUTDIR/energy_bf16.json"
    $BENCH --model "$MODEL" --prompt-tokens 128 --gen 1024 \
           --warmup 1 --reps 1 --seed 42 --precision bf16 --out "$eout"
    kill "$pm_pid" 2>/dev/null || true
    wait "$pm_pid" 2>/dev/null || true
    avgw="$(python3 - "$pmf" <<'PY'
import re, sys
# Average the "GPU Power" and "CPU Power" (mW) samples across the file.
gpu=[]; cpu=[]
for line in open(sys.argv[1], errors='ignore'):
    m=re.search(r'GPU Power:\s*([\d.]+)\s*mW', line)
    if m: gpu.append(float(m.group(1)))
    m=re.search(r'CPU Power:\s*([\d.]+)\s*mW', line)
    if m: cpu.append(float(m.group(1)))
vals=[x for x in (gpu+cpu) if x>0]
print(f"{sum(vals)/len(vals)/1000.0:.2f}" if vals else "0")
PY
)"
    rm -f "$pmf"
    python3 - "$eout" "$avgw" <<'PY'
import json, sys
p, avgw = sys.argv[1], float(sys.argv[2])
d = json.load(open(p))
r = d["results"]
active = r.get("active_seconds", 0.0)
tokens = r.get("generated_tokens", 0)
joules = avgw * active
r["energy"] = {
    "avg_w": avgw,
    "joules": round(joules, 2),
    "j_per_token": round(joules / tokens, 4) if tokens else 0.0,
}
json.dump(d, open(p, "w"), indent=2)
print(f"  energy: {avgw:.2f} W -> {joules:.2f} J, {r['energy']['j_per_token']:.4f} J/token")
PY
  fi
fi

echo
echo "== summary =="
python3 - "$OUTDIR" <<'PY'
import json, glob, os, sys
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: j = json.load(open(f))
    except Exception: continue
    if j.get("schema") != "baremetal.bench/v1": continue
    m, c, r = j["meta"], j["config"], j["results"]
    rows.append((os.path.basename(f), m["precision"], m["quant"],
                 c["prompt_tokens"], c["gen_tokens"],
                 r["prefill"]["tok_s"]["median"], r["decode"]["tok_s"]["median"],
                 r["ttft_ms"]["p50"], r["itl_ms"]["p50"]))
if not rows:
    print("(no results)"); raise SystemExit
print(f"{'file':<22}{'prec':<6}{'q':<5}{'Np':>5}{'Ng':>5}{'pref t/s':>10}{'dec t/s':>10}{'ttft ms':>9}{'itl ms':>8}")
for x in rows:
    print(f"{x[0]:<22}{x[1]:<6}{x[2]:<5}{x[3]:>5}{x[4]:>5}{x[5]:>10.1f}{x[6]:>10.1f}{x[7]:>9.2f}{x[8]:>8.2f}")
PY
