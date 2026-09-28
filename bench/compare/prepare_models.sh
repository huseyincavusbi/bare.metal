#!/usr/bin/env bash
# bench/compare/prepare_models.sh -- generate engine-specific model variants
# for a HuggingFace model directory.
#
# Produces (skipping anything that already exists):
#   models/<stem>-f16.gguf          (llama.cpp base + F16 datapoint)
#   models/<stem>-q8_0.gguf         (llama.cpp Q8_0)
#   models/<stem>-q4_0.gguf         (llama.cpp Q4_0)
#   models/<stem>-mlx/              (MLX bfloat16)
#   models/<stem>-mlx-fp16/         (MLX float16)
#   models/<stem>-mlx-q8/           (MLX 8-bit group quant)
#   models/<stem>-mlx-q4/           (MLX 4-bit group quant)
#
# Requires (see AGENTS / benchmark notes):
#   $BENCH_VENV_HOME/gguf-venv  with llama.cpp convert requirements
#   $BENCH_VENV_HOME/llama.cpp  the llama.cpp checkout (convert script)
#   $BENCH_VENV_HOME/bench-venv with mlx-lm
#   llama-quantize on PATH (brew install llama.cpp)
#
# Usage: prepare_models.sh <stem> [hf-dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

STEM="${1:-}"
HF="${2:-models/$STEM}"
CACHE="${BENCH_VENV_HOME:-$HOME/.cache/baremetal}"
CONVERT_PY="$CACHE/gguf-venv/bin/python"
MLX_PY="$CACHE/bench-venv/bin/python"
LLAMACPP="${LLAMACPP_DIR:-$CACHE/llama.cpp}"

[ -n "$STEM" ] || { echo "usage: $0 <stem> [hf-dir]" >&2; exit 2; }
[ -f "$HF/config.json" ] || { echo "no HF model at $HF" >&2; exit 1; }
[ -x "$CONVERT_PY" ] || { echo "missing $CONVERT_PY" >&2; exit 1; }
[ -x "$MLX_PY" ]     || { echo "missing $MLX_PY" >&2; exit 1; }
[ -f "$LLAMACPP/convert_hf_to_gguf.py" ] || { echo "missing $LLAMACPP/convert_hf_to_gguf.py" >&2; exit 1; }
mkdir -p models

echo "== $STEM : GGUF f16 =="
if [ ! -f "models/$STEM-f16.gguf" ]; then
  "$CONVERT_PY" "$LLAMACPP/convert_hf_to_gguf.py" "$HF" --outtype f16 --outfile "models/$STEM-f16.gguf"
else
  echo "  exists: models/$STEM-f16.gguf"
fi

if command -v llama-quantize >/dev/null 2>&1; then
  for q in q8_0 q4_0; do
    echo "== $STEM : GGUF $q =="
    if [ ! -f "models/$STEM-$q.gguf" ]; then
      qt="$(echo "$q" | tr '[:lower:]' '[:upper:]')"
      llama-quantize "models/$STEM-f16.gguf" "models/$STEM-$q.gguf" "$qt" >/dev/null
    else
      echo "  exists: models/$STEM-$q.gguf"
    fi
  done
else
  echo "  llama-quantize not found; skipping quants" >&2
fi

mlx_convert() {  # <outdir> <extra args...>
  local out="$1"; shift
  if [ -d "$out" ]; then echo "  exists: $out"; return; fi
  echo "== $STEM : MLX $out =="
  "$MLX_PY" -m mlx_lm convert --hf-path "$HF" --mlx-path "$out" "$@" 2>&1 | grep -viE "warning|it/s\]|B/s\]|Fetching|^\s*$" | tail -2
}

mlx_convert "models/$STEM-mlx"      --dtype bfloat16
mlx_convert "models/$STEM-mlx-fp16" --dtype float16
mlx_convert "models/$STEM-mlx-q8"   -q --q-bits 8
mlx_convert "models/$STEM-mlx-q4"   -q --q-bits 4

echo
echo "== $STEM : variants =="
ls -1d models/"$STEM"* | sed 's/^/  /'
