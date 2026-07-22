"""Export HuggingFace tokenizer to bare.metal binary format.

Reads tokenizer.json directly (no HF dependency) and assigns BPE merge
scores based on merge rank: earlier merges get higher scores so the
BPE loop merges them first. Non-merge tokens (special tokens, base
byte tokens) get -1e20 so they are never selected as merge targets.
"""
import sys, struct, os, json

model_dir = sys.argv[1]
out_path = sys.argv[2]

tok_path = os.path.join(model_dir, "tokenizer.json")
with open(tok_path) as f:
    tok = json.load(f)

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)

vocab = tok["model"]["vocab"]
merges = tok["model"]["merges"]
id_to_token = {v: k for k, v in vocab.items()}
V = cfg["vocab_size"]

merge_scores = {}
for i, merge in enumerate(merges):
    parts = merge.split(" ")
    if len(parts) < 2:
        continue
    merged = parts[0] + parts[1]
    if merged in vocab:
        merge_scores[vocab[merged]] = float(len(merges) - i)

max_len = 0
for i in range(V):
    s = id_to_token.get(i, "")
    max_len = max(max_len, len(s.encode("utf-8")))

with open(out_path, "wb") as f:
    f.write(struct.pack("i", max_len))
    for i in range(V):
        s = id_to_token.get(i, "")
        b = s.encode("utf-8")
        if i in merge_scores:
            score = merge_scores[i]
        else:
            score = -1e20
        f.write(struct.pack("fi", score, len(b)))
        f.write(b)

print("Exported {}: V={} n_merges={} max_len={}".format(out_path, V, len(merges), max_len))