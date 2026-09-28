#!/usr/bin/env bash
# bench/run_benchmark.sh -- single benchmark driver.
#
# Runs every step (model prep, cross-engine runs, bare.metal sweep, energy)
# for every supported model by default, then aggregates everything into one
# timestamped run folder:
#
#   bench/benchmarks/<MM-DD-YYYY-HHMM>-<name>/
#     run.json                          run metadata (version, host, settings)
#     grand.json                        all models x engines x formats (full raw)
#     models/<stem>.json                per-model aggregate
#     engines/<engine>.json             per-engine aggregate
#     raw/<stem>/<engine>__<fmt>.json    individual baremetal.bench/v2 runs
#     raw/<stem>/sweep__<cfg>.json      bare.metal context-scaling runs
#     energy/                           raw powermetrics logs
#
# Real Apple Silicon hardware only: GitHub CI runners are paravirtual.
#
# Usage:
#   bench/run_benchmark.sh [options]
#
# Options:
#   --models LIST    comma-separated stems           (default: all HF dirs in models/)
#   --stages LIST    prep,cross,sweep,aggregate      (default: all)
#   --formats LIST   size classes: bf16,fp16,q8,q4   (default: all; mapped per
#                    engine, unsupported classes skipped)
#   --prompt-tokens N  cross-engine prompt cap        (default: 128)
#   --gen N          tokens to generate               (default: 128)
#   --reps N         measured reps                    (default: 3)
#   --name TAG       folder label                     (default: benchmark)
#   --out DIR        base output dir                  (default: bench/benchmarks)
#   --no-energy      skip powermetrics capture
#   --quick          smoke: prompt 32, gen 8, reps 1, no energy, skip sweep
#   --dry-run        print the plan and exit
#
# Environment:
#   BENCH_VENV_HOME  dir with bench-venv + gguf-venv + llama.cpp (default ~/.cache/baremetal)
#   PYTHON           python for the adapters        (default bench-venv python)
#   PEAK_GBPS / PEAK_GFLOPPS  device peaks for % of peak (default M4: 120 / 4200)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CACHE="${BENCH_VENV_HOME:-$HOME/.cache/baremetal}"
BENCH_PY="${PYTHON:-$CACHE/bench-venv/bin/python}"
CONVERT_PY="$CACHE/gguf-venv/bin/python"
LLAMACPP_DIR="${LLAMACPP_DIR:-$CACHE/llama.cpp}"
BENCH="./build/bench/bench"

MODELS=""
STAGES="prep,cross,sweep,aggregate"
FORMATS="bf16,fp16,q8,q4"
PROMPT=128
GEN=128
REPS=3
NAME="benchmark"
OUT="bench/benchmarks"
ENERGY=1
QUICK=0
DRY=0
PEAK_GBPS="${PEAK_GBPS:-120}"
PEAK_GFLOPPS="${PEAK_GFLOPPS:-4200}"

usage() { sed -n '2,42p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --models)        MODELS="$2"; shift 2;;
    --stages)        STAGES="$2"; shift 2;;
    --formats)       FORMATS="$2"; shift 2;;
    --prompt-tokens) PROMPT="$2"; shift 2;;
    --gen)           GEN="$2"; shift 2;;
    --reps)          REPS="$2"; shift 2;;
    --name)          NAME="$2"; shift 2;;
    --out)           OUT="$2"; shift 2;;
    --quick)         QUICK=1; shift;;
    --dry-run)       DRY=1; shift;;
    --no-energy)     ENERGY=0; shift;;
    -h|--help)       usage; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

if [ "$QUICK" -eq 1 ]; then
  GEN=8; REPS=1; ENERGY=0
  [ "$STAGES" = "prep,cross,sweep,aggregate" ] && STAGES="prep,cross,aggregate"
fi
[ "$ENERGY" -eq 1 ] && [ "$QUICK" -eq 1 ] && ENERGY=0

in_list() { case ",$1," in *",$2,"*) return 0;; *) return 1;; esac; }
want_stage() { in_list "$STAGES" "$1"; }
want_fmt()   { in_list "$FORMATS" "$1"; }

# ---- engine/format mapping --------------------------------------------------
llama_file_for() {
  case "$2" in
    bf16) echo "models/$1-bf16.gguf";;
    fp16) echo "models/$1-f16.gguf";;
    q8)   echo "models/$1-q8_0.gguf";;
    q4)   echo "models/$1-q4_0.gguf";;
    *)    echo "";;
  esac
}
mlx_dir_for() {
  case "$2" in
    bf16) echo "models/$1-mlx";;
    fp16) echo "models/$1-mlx-fp16";;
    q8)   echo "models/$1-mlx-q8";;
    q4)   echo "models/$1-mlx-q4";;
    *)    echo "";;
  esac
}
bare_args_for() {
  case "$1" in
    bf16) echo "--precision bf16";;
    fp16) echo "--precision fp16";;
    fp32) echo "--precision fp32";;
    q8)   echo "--quant q8";;
    q4)   echo "--quant q4";;
    *)    echo "";;
  esac
}
mps_dtype_for() {
  case "$1" in
    bf16) echo "bf16";;
    fp16) echo "fp16";;
    fp32) echo "fp32";;
    *)    echo "";;
  esac
}

# ---- model discovery --------------------------------------------------------
if [ -z "$MODELS" ]; then
  MODELS=""
  for d in models/*/; do
    b="${d%/}"; b="${b##*/}"
    case "$b" in *-mlx*) continue;; esac
    if [ -f "${d}config.json" ] && [ -f "${d}tokenizer.json" ]; then
      MODELS="${MODELS:+$MODELS,}$b"
    fi
  done
fi
[ -n "$MODELS" ] || { echo "no models found under models/" >&2; exit 1; }

RUN_DIR="$OUT/$(date +%m-%d-%Y-%H%M)-$NAME"
mkdir -p "$RUN_DIR/raw" "$RUN_DIR/energy" "$RUN_DIR/models" "$RUN_DIR/engines"

echo "run:     $RUN_DIR"
echo "models:  $MODELS"
echo "stages:  $STAGES"
echo "formats: $FORMATS"
echo "config:  prompt<=$PROMPT gen=$GEN reps=$REPS energy=$ENERGY"
echo

if [ "$DRY" -eq 1 ]; then
  for stem in ${MODELS//,/ }; do
    echo "model: $stem"
    for fmt in ${FORMATS//,/ }; do
      [ -n "$(bare_args_for "$fmt")" ] && echo "  bare.metal  $fmt"
      [ -n "$(llama_file_for "$stem" "$fmt")" ] && echo "  llama.cpp   $fmt -> $(llama_file_for "$stem" "$fmt")"
      [ -n "$(mlx_dir_for "$stem" "$fmt")" ] && echo "  mlx         $fmt -> $(mlx_dir_for "$stem" "$fmt")"
      [ -n "$(mps_dtype_for "$fmt")" ] && echo "  torch-mps   $fmt"
    done
  done
  echo "(dry run; nothing executed)"
  exit 0
fi

# ---- energy helpers ---------------------------------------------------------
energy_detect() {
  PM_BIN="$(command -v powermetrics || true)"
  HAVE_ENERGY=0
  [ -n "$PM_BIN" ] || { echo "energy: powermetrics not found; skipping" >&2; return; }
  if sudo -n "$PM_BIN" -n 1 -i 50 -s gpu_power -o /dev/null >/dev/null 2>&1; then
    HAVE_ENERGY=1
  else
    echo "energy: no passwordless sudo for powermetrics; skipping" >&2
    echo "  grant: echo \"$(id -un) ALL=(root) NOPASSWD: $PM_BIN\" | sudo tee /etc/sudoers.d/powermetrics >/dev/null && sudo chmod 440 /etc/sudoers.d/powermetrics" >&2
  fi
}

_energy_merge() {
  python3 - "$1" "$2" <<'PY' || true
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

blocks, cur = [], new_block()
for line in open(pm_path, errors='ignore'):
    m = re.search(r'\*\*\* Sampled system activity \(([^)]*)\)', line)
    if m:
        if cur["ts"] is not None:
            blocks.append(cur)
        cur = new_block(); cur["ts"] = parse_ts(m.group(1)); continue
    if cur["ts"] is None:
        continue
    if "Average" in line or "(avg" in line:
        continue
    for (key, rx, scale) in PATTERNS:
        mm = re.search(rx, line)
        if mm:
            cur["vals"][key].append(float(mm.group(1)) * scale)
if cur["ts"] is not None:
    blocks.append(cur)

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

cpu_w, gpu_w, ane_w = mean("avg_cpu_w"), mean("avg_gpu_w"), mean("avg_ane_w")
total_w = cpu_w + gpu_w + ane_w
if total_w == 0.0:
    total_w = mean("combined_w")
energy["avg_cpu_w"], energy["avg_gpu_w"], energy["avg_ane_w"] = round(cpu_w, 2), round(gpu_w, 2), round(ane_w, 2)
energy["avg_w"] = round(total_w, 2)
energy["powermetrics_log"] = os.path.basename(out_path).replace(".json", ".powermetrics.txt")

active = r.get("active_seconds", 0.0)
reps = c.get("reps", 1)
joules = total_w * active
energy["joules"] = round(joules, 2)
if c.get("gen_tokens", 0) > 8:
    dec_tokens = (c["gen_tokens"] - 1) * reps
    energy["j_per_decode_token"] = round(joules / dec_tokens, 4) if dec_tokens else 0.0
gen = r.get("generated_tokens", 0)
energy["j_per_token"] = round(joules / gen, 4) if gen else 0.0
r["energy"] = energy
json.dump(d, open(out_path, "w"), indent=2)
print(f"  energy: {total_w:.2f} W, {joules:.1f} J, {len(sel)} samples, {energy.get('gpu_freq_mhz','?')} MHz")
PY
}

energy_exec() {
  local out="$1"; shift
  local pmf; pmf="$(mktemp)"
  local pmlog="$RUN_DIR/energy/$(basename "$out" .json).powermetrics.txt"
  sudo powermetrics --samplers cpu_power,gpu_power,thermal,ane_power --show-process-gpu \
       -i 250 -o "$pmf" >/dev/null 2>&1 &
  local pm_pid=$!
  sleep 1
  local rc=0
  "$@" >/dev/null || rc=$?
  kill "$pm_pid" 2>/dev/null || true
  wait "$pm_pid" 2>/dev/null || true
  _energy_merge "$out" "$pmf"
  cp -f "$pmf" "$pmlog" 2>/dev/null || true
  rm -f "$pmf"
  return "$rc"
}

HAVE_ENERGY=0
[ "$ENERGY" -eq 1 ] && energy_detect

# ---- stages -----------------------------------------------------------------
prep_model() {
  local stem="$1" hf="models/$1"
  echo "  prep $stem: gguf"
  for t in f16 bf16; do
    [ -f "models/$stem-$t.gguf" ] || "$CONVERT_PY" "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$hf" --outtype "$t" --outfile "models/$stem-$t.gguf" >/dev/null 2>&1
  done
  if command -v llama-quantize >/dev/null 2>&1; then
    for q in q8_0 q4_0; do
      [ -f "models/$stem-$q.gguf" ] || llama-quantize "models/$stem-f16.gguf" "models/$stem-$q.gguf" "$(echo "$q" | tr '[:lower:]' '[:upper:]')" >/dev/null 2>&1
    done
  fi
  echo "  prep $stem: mlx"
  mlx_conv() { local o="$1"; shift; [ -d "$o" ] || "$BENCH_PY" -m mlx_lm convert --hf-path "$hf" --mlx-path "$o" "$@" >/dev/null 2>&1; }
  mlx_conv "models/$stem-mlx" --dtype bfloat16
  mlx_conv "models/$stem-mlx-fp16" --dtype float16
  mlx_conv "models/$stem-mlx-q8" -q --q-bits 8
  mlx_conv "models/$stem-mlx-q4" -q --q-bits 4
}

run_engine() {  # <out> <cmd...>
  local out="$1"; shift
  mkdir -p "$(dirname "$out")"
  if [ "$HAVE_ENERGY" -eq 1 ]; then
    energy_exec "$out" "$@"
  else
    "$@" >/dev/null || echo "  FAILED: $(basename "$out")"
  fi
}

cross_model() {
  local stem="$1"
  echo "== cross $stem =="
  for fmt in ${FORMATS//,/ }; do
    local out="$RUN_DIR/raw/$stem"
    # bare.metal
    if [ -n "$(bare_args_for "$fmt")" ]; then
      # shellcheck disable=SC2086
      run_engine "$out/bare.metal__$fmt.json" "$BENCH" --model "models/$stem" --seed 42 \
        --prompt-tokens "$PROMPT" --raw-prompt --gen "$GEN" --warmup 1 --reps "$REPS" \
        --peak-gbps "$PEAK_GBPS" --peak-gflops "$PEAK_GFLOPPS" $(bare_args_for "$fmt") \
        --out "$out/bare.metal__$fmt.json"
    fi
    # llama.cpp
    local gf; gf="$(llama_file_for "$stem" "$fmt")"
    if [ -n "$gf" ] && [ -f "$gf" ] && command -v llama-bench >/dev/null 2>&1; then
      local lprec; case "$fmt" in bf16) lprec=BF16;; fp16) lprec=F16;; q8) lprec=Q8_0;; q4) lprec=Q4_0;; esac
      run_engine "$out/llama.cpp__$fmt.json" python3 bench/compare/bench_llamacpp.py \
        --gguf "$gf" --precision "$lprec" --gen "$GEN" --reps "$REPS" \
        --peak-gflops "$PEAK_GFLOPPS" --peak-gbps "$PEAK_GBPS" \
        --out "$out/llama.cpp__$fmt.json"
    fi
    # MLX
    local md; md="$(mlx_dir_for "$stem" "$fmt")"
    if [ -n "$md" ] && [ -d "$md" ] && "$BENCH_PY" -c "import mlx_lm" 2>/dev/null; then
      run_engine "$out/mlx__$fmt.json" "$BENCH_PY" bench/compare/bench_mlx.py \
        --model "$md" --gen "$GEN" --reps "$REPS" \
        --peak-gflops "$PEAK_GFLOPPS" --peak-gbps "$PEAK_GBPS" \
        --out "$out/mlx__$fmt.json"
    fi
    # torch-mps
    local dt; dt="$(mps_dtype_for "$fmt")"
    if [ -n "$dt" ] && "$BENCH_PY" -c "import torch" 2>/dev/null; then
      run_engine "$out/torch-mps__$fmt.json" "$BENCH_PY" bench/compare/bench_mps.py \
        --model "models/$stem" --dtype "$dt" --gen "$GEN" --reps "$REPS" \
        --peak-gflops "$PEAK_GFLOPPS" --peak-gbps "$PEAK_GBPS" \
        --out "$out/torch-mps__$fmt.json"
    fi
  done
}

sweep_model() {
  local stem="$1"
  echo "== sweep $stem (bare.metal) =="
  for cfg in 128:bf16 128:fp16 128:q8 128:q4 512:bf16 1024:bf16; do
    local p="${cfg%%:*}" pr="${cfg##*:}"
    local out="$RUN_DIR/raw/$stem/sweep__p${p}_${pr}.json"
    # shellcheck disable=SC2086
    if [ "$HAVE_ENERGY" -eq 1 ]; then
      energy_exec "$out" "$BENCH" --model "models/$stem" --seed 42 \
        --prompt-tokens "$p" --gen "$GEN" --warmup 1 --reps "$REPS" \
        --peak-gbps "$PEAK_GBPS" --peak-gflops "$PEAK_GFLOPPS" $(bare_args_for "$pr") --out "$out"
    else
      "$BENCH" --model "models/$stem" --seed 42 \
        --prompt-tokens "$p" --gen "$GEN" --warmup 1 --reps "$REPS" \
        --peak-gbps "$PEAK_GBPS" --peak-gflops "$PEAK_GFLOPPS" $(bare_args_for "$pr") --out "$out" >/dev/null
    fi
  done
}

make bench >/dev/null 2>&1 || { echo "build failed" >&2; exit 1; }

for stem in ${MODELS//,/ }; do
  want_stage prep  && prep_model "$stem"
done
for stem in ${MODELS//,/ }; do
  want_stage cross && cross_model "$stem"
done
for stem in ${MODELS//,/ }; do
  want_stage sweep && sweep_model "$stem"
done
if want_stage aggregate; then
  echo
  "$BENCH_PY" bench/compare/collect.py --run-dir "$RUN_DIR" --settings \
      "prompt_cap=$PROMPT,gen=$GEN,reps=$REPS,formats=$FORMATS,stages=$STAGES" || \
    echo "aggregate failed" >&2
fi
echo
echo "done: $RUN_DIR"
