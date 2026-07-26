import torch, numpy as np, os
from transformers import AutoModelForCausalLM, AutoTokenizer
md='data/smollm2-135m'
tok=AutoTokenizer.from_pretrained(md)
m=AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float32).eval()
ids=torch.tensor([[6403, 1980, 253, 655]])   # "Once upon a time"
with torch.no_grad():
    logits=m(ids).logits[0].numpy().astype(np.float32)   # [S, V]
print('logits', logits.shape)
os.makedirs('test/grad', exist_ok=True)
logits.tofile('test/grad/fwd_train_logits.bin')
print('S=', logits.shape[0], 'V=', logits.shape[1])
