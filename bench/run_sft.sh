#!/usr/bin/env bash
# bench/run_sft.sh -- run and publish a focused JSONL SFT benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS="smollm2-135m,Llama-3.2-1B"
DATA=""
STEPS=200
SEQ=128
LR=3e-4
VAL_SPLIT=10
NAME="sft-dolly-v1"
PUBLISH=1
RESULTS="${BENCH_RESULTS_DIR:-$HOME/.cache/baremetal/benchmarks-repo}"

while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --seq-len) SEQ="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --val-split) VAL_SPLIT="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --no-publish) PUBLISH=0; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$DATA" ] || { echo "usage: $0 --data FILE [--models ...]" >&2; exit 2; }
[ -f "$DATA" ] || { echo "missing data: $DATA" >&2; exit 2; }

make bench >/dev/null

STAMP="$(date +%m-%d-%Y-%H%M)"
RUN_DIR="bench/benchmarks/${STAMP}-${NAME}"
mkdir -p "$RUN_DIR/raw" "$RUN_DIR/adapters"
echo "run: $RUN_DIR"

for stem in ${MODELS//,/ }; do
  model="models/$stem"
  [ -d "$model" ] || { echo "missing model: $model" >&2; exit 2; }
  lr="$LR"
  case "$stem" in *Llama*) lr=1e-4;; esac
  out="$RUN_DIR/raw/$stem/train__bf16_sft.json"
  adapter="$RUN_DIR/adapters/${stem}.bin"
  mkdir -p "$(dirname "$out")"
  echo "== sft $stem lr=$lr steps=$STEPS seq=$SEQ =="
  ./build/bench/train_bench \
    --model "$model" \
    --sft-data "$DATA" \
    --template auto \
    --val-split "$VAL_SPLIT" \
    --steps "$STEPS" \
    --seq-len "$SEQ" \
    --lr "$lr" \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-targets all \
    --adapter-out "$adapter" \
    --out "$out"
done

python3 bench/compare/collect.py --run-dir "$RUN_DIR" \
  --settings "stages=sft,sft_steps=$STEPS,sft_seq=$SEQ,dataset=$(basename "$DATA"),val_split=$VAL_SPLIT"

if [ "$PUBLISH" -eq 1 ]; then
  if [ ! -d "$RESULTS/.git" ]; then
    gh repo clone huseyincavusbi/bare.metal-benchmarks "$RESULTS" >/dev/null
  fi
  if [ -n "$(git -C "$RESULTS" status --porcelain)" ]; then
    echo "results checkout dirty; skipping publish" >&2
    exit 0
  fi
  git -C "$RESULTS" pull --ff-only -q
  dst="$RESULTS/runs/$(basename "$RUN_DIR")"
  rm -rf "$dst"; mkdir -p "$dst"
  cp -f "$RUN_DIR/run.json" "$RUN_DIR/grand.json" "$RUN_DIR/summary.md" "$dst/"
  cp -R "$RUN_DIR/models" "$dst/"
  python3 "$RESULTS/tools/index.py" >/dev/null
  git -C "$RESULTS" add -A
  git -C "$RESULTS" commit -q -m "bench: $(basename "$RUN_DIR")"
  git -C "$RESULTS" push -q origin HEAD
  echo "published $(basename "$RUN_DIR")"
fi

echo "done: $RUN_DIR"
