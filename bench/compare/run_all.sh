#!/usr/bin/env bash
# bench/compare/run_all.sh -- run the inference benchmark across engines
# sequentially and print a comparison.
#
# Engines that are not installed are skipped with a hint. Real hardware only
# (CI runners are paravirtual). Models must share the same tokenizer/weights.
#
# Usage:
#   bench/compare/run_all.sh [--model DIR] [--mlx-model ID] [--gguf PATH]
#                            [--prompt-tokens N] [--gen N] [--reps N]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-models/smollm2-135m}"
MLX_MODEL="${MLX_MODEL:-models/smollm2-135m-mlx}"
GGUF="${GGUF:-models/smollm2-135m-bf16.gguf}"
PROMPT_TOKENS=128
GEN=32
REPS=5
PREC=bf16
PYTHON="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2;;
    --mlx-model)     MLX_MODEL="$2"; shift 2;;
    --gguf)          GGUF="$2"; shift 2;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2;;
    --gen)           GEN="$2"; shift 2;;
    --reps)          REPS="$2"; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

OUT="bench/compare/results/$(date +%m-%d-%Y)"
PARTS="$(mktemp -d)"
mkdir -p "$OUT"
echo "run dir: $OUT (single file)"
echo "host:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
echo

echo "== bare.metal =="
make bench >/dev/null 2>&1
./build/bench/bench --model "$MODEL" --prompt-tokens "$PROMPT_TOKENS" --raw-prompt --gen "$GEN" \
    --warmup 1 --reps "$REPS" --precision "$PREC" --out "$PARTS/baremetal_${PREC}.json" \
  || echo "  bare.metal bench failed"

echo "== llama.cpp =="
if command -v llama-bench >/dev/null 2>&1; then
  if [ -z "$GGUF" ] || [ ! -f "$GGUF" ]; then
    echo "  no GGUF available (set GGUF=path or --gguf); skipping"
  else
    python3 bench/compare/bench_llamacpp.py --gguf "$GGUF" \
        --gen "$GEN" --reps "$REPS" --out "$PARTS/llamacpp.json" || echo "  llama.cpp bench failed"
  fi
else
  echo "  llama-bench not found (brew install llama.cpp); skipping"
fi

echo "== MLX =="
if "$PYTHON" -c "import mlx_lm" 2>/dev/null; then
  "$PYTHON" bench/compare/bench_mlx.py --model "$MLX_MODEL" \
      --gen "$GEN" --reps "$REPS" --out "$PARTS/mlx.json" || echo "  MLX bench failed"
else
  echo "  mlx-lm not installed in $PYTHON (uv pip install mlx-lm); skipping"
fi

echo "== PyTorch MPS =="
if "$PYTHON" -c "import torch" 2>/dev/null; then
  "$PYTHON" bench/compare/bench_mps.py --model "$MODEL" \
      --gen "$GEN" --reps "$REPS" --out "$PARTS/mps.json" || echo "  MPS bench failed"
else
  echo "  torch not installed in $PYTHON; skipping"
fi

echo
"$PYTHON" bench/compare/collect.py --outdir "$OUT" --benchmark cross-engine \
    --date "$(basename "$OUT")" "$PARTS"
rm -rf "$PARTS"
