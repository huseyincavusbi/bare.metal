import sys, struct, os, json
from transformers import AutoTokenizer

model_dir = sys.argv[1]
out_path = sys.argv[2]

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)

tok = AutoTokenizer.from_pretrained(model_dir)
V = cfg["vocab_size"]

vocab = tok.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

max_len = 0
for i in range(V):
    s = id_to_token.get(i, f"<UNK{i}>")
    max_len = max(max_len, len(s.encode("utf-8")))

with open(out_path, "wb") as f:
    f.write(struct.pack("i", max_len))
    for i in range(V):
        s = id_to_token.get(i, f"<UNK{i}>")
        b = s.encode("utf-8")
        score = float(V - i)
        f.write(struct.pack("fi", score, len(b)))
        f.write(b)

print("Exported {}: V={} max_len={}".format(out_path, V, max_len))
