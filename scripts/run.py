#!/usr/bin/env python3
"""Orchestrator: tokenize via HF, invoke baremetal run-tokens, decode output."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = REPO_ROOT / "build" / "baremetal"


def write_token_ids(ids, path: Path) -> None:
    n = len(ids)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        if n:
            f.write(struct.pack(f"<{n}i", *ids))


def read_token_ids(path: Path):
    raw = path.read_bytes()
    if len(raw) < 4:
        return []
    (n,) = struct.unpack("<i", raw[:4])
    if n < 0 or n * 4 > len(raw) - 4:
        return []
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}i", raw[4:4 + 4 * n]))


def main():
    p = argparse.ArgumentParser(
        description="Run bare.metal inference with HF tokenization",
    )
    p.add_argument("model_dir", type=Path, help="HuggingFace model directory")
    p.add_argument("prompt", type=str, help="Text prompt to generate from")
    p.add_argument("--steps", type=int, default=256, help="Max new tokens (default: 256)")
    p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    p.add_argument("--top-k", type=int, default=40, help="Top-k sampling (default: 40)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling (default: 0.9)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed, 0=time-based (default: 0)")
    p.add_argument("--binary", type=Path, default=DEFAULT_BIN, help=f"baremetal binary (default: {DEFAULT_BIN})")
    p.add_argument("--keep-files", action="store_true", help="Keep prompt_ids.bin / output_ids.bin in cwd")
    p.add_argument("--prompt-ids-out", type=Path, default=None, help="Override path for prompt_ids.bin")
    p.add_argument("--output-ids-out", type=Path, default=None, help="Override path for output_ids.bin")
    args = p.parse_args()

    if not args.model_dir.is_dir():
        print(f"error: model dir not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.binary.exists():
        print(f"error: baremetal binary not found: {args.binary}", file=sys.stderr)
        print(f"hint: run 'make baremetal' first", file=sys.stderr)
        sys.exit(1)

    print(f"[run] model:    {args.model_dir}")

    print(f"[run] tokenizer: {args.model_dir}")
    try:
        tok = AutoTokenizer.from_pretrained(str(args.model_dir))
        if not tok:
            tok = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=False)
    prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
    print(f"[run] prompt:    {args.prompt!r}")
    print(f"[run] prompt_ids ({len(prompt_ids)}): {prompt_ids}")

    if args.prompt_ids_out and args.output_ids_out:
        prompt_path = args.prompt_ids_out
        output_path = args.output_ids_out
        write_token_ids(prompt_ids, prompt_path)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="baremetal_run_")
        tmp = Path(tmp_ctx.name)
        prompt_path = tmp / "prompt_ids.bin"
        output_path = tmp / "output_ids.bin"
        write_token_ids(prompt_ids, prompt_path)

    cmd = [
        str(args.binary), "run-tokens",
        str(args.model_dir),
        str(prompt_path),
        str(output_path),
        "--steps", str(args.steps),
        "--temp", str(args.temp),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--seed", str(args.seed),
    ]
    print(f"[run] cmd:      {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        if tmp_ctx is not None and not args.keep_files:
            tmp_ctx.cleanup()
        print(f"[run] baremetal exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    if not output_path.exists():
        if tmp_ctx is not None and not args.keep_files:
            tmp_ctx.cleanup()
        print(f"[run] error: C side did not produce {output_path}", file=sys.stderr)
        sys.exit(1)

    if tmp_ctx is not None and args.keep_files:
        kept_p = Path("prompt_ids.bin").resolve()
        kept_o = Path("output_ids.bin").resolve()
        kept_p.write_bytes(prompt_path.read_bytes())
        kept_o.write_bytes(output_path.read_bytes())
        print(f"[run] kept:     {kept_p}")
        print(f"[run] kept:     {kept_o}")

    gen_ids = read_token_ids(output_path)
    print(f"[run] generated_ids ({len(gen_ids)}): {gen_ids}")
    text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"[run] generated_text:\n{text}")

    if tmp_ctx is not None and not args.keep_files:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
