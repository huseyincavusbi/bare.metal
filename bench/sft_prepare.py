#!/usr/bin/env python3
"""Convert a deterministic Dolly-15k subset to a messages JSONL file."""

import argparse
import json
import os
import random
import urllib.request

URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft-data/dolly_600.jsonl")
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", default="")
    args = ap.parse_args()

    if args.source:
        lines = open(args.source, encoding="utf-8").read().splitlines()
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            lines = r.read().decode("utf-8").splitlines()

    rows = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        instruction = (d.get("instruction") or "").strip()
        response = (d.get("response") or "").strip()
        context = (d.get("context") or "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else instruction + "\n\nContext:\n" + context
        rows.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": response},
            ]
        })

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.count]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
