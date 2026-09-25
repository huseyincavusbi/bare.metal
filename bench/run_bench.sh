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
#   --model DIR      model directory            (default: models/smollm2-135m)
#   --out DIR        base results directory     (default: bench/results)
#   --reps N         measured reps per config  (default: 5)
#   --warmup N       warmup reps                (default: 2)
#   --gen N          tokens to generate         (default: 128)
#   --quick          single config, fewer reps  (smoke)
#   --no-sweep       only the base config
#   --no-energy      skip powermetrics energy capture

set -euo pipefail

MODEL="${MODEL:-models/smollm2-135m}"
OUTDIR="${OUTDIR:-bench/results}"
REPS=5
WARMUP=2
GEN=128
SWEEP=1
QUICK=0
ENERGY=1
PEAK_GBPS="${PEAK_GBPS:-120}"
PEAK_GFLOPPS="${PEAK_GFLOPPS:-4200}"

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
  CONFIGS="128:bf16 128:fp16 128:q8 128:q4 512:bf16 1024:bf16 2048:bf16"
else
  CONFIGS="128:bf16"
fi
[ "$QUICK" -eq 1 ] && CONFIGS="64:bf16"

run_one() {
  local prompt="$1" prec="$2" tag="$3" extra="$4"
  local out="$RUN_DIR/${tag}.json"
  # shellcheck disable=SC2086
  $BENCH --model "$MODEL" --prompt-tokens "$prompt" --gen "$GEN" \
         --warmup "$WARMUP" --reps "$REPS" --seed 42 \
         --peak-gbps "$PEAK_GBPS" --peak-gflops "$PEAK_GFLOPPS" $extra --out "$out"
  echo "  wrote $out"
}

for cfg in $CONFIGS; do
  prompt="${cfg%%:*}"; prec="${cfg##*:}"
  extra=""
  if [ "$prec" = "q8" ] || [ "$prec" = "q4" ]; then extra="--quant $prec"; else extra="--precision $prec"; fi
  echo "== prompt=$prompt precision=$prec =="
  run_one "$prompt" "$prec" "p${prompt}_${prec}" "$extra"
done

# ---- energy capture (default on; powermetrics needs passwordless sudo) -----
HAVE_ENERGY=0
source "$ROOT/bench/energy.sh"
if [ "$ENERGY" -eq 1 ]; then energy_detect; fi

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
