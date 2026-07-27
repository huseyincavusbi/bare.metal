import torch, numpy as np, os
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
md='data/smollm2-135m'
tok=AutoTokenizer.from_pretrained(md)
m=AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float32)
# 10 training sequences (S=8), from a fixed TinyStories-ish text
text = "Once upon a time there was a little girl named Lily who lived in a big house with her family and friends. One sunny day Lily decided to teach her students about the number two. She began by saying hello to the class and then she told them a long story about a bird and a tree and a river and a mountain and a cat and a dog and a fish and a star and a moon and a sun and a cloud and a rain and a snow"
ids = tok.encode(text, return_tensors='pt')[0]   # length ~20
S = 8
seqs = [ids[i:i+S+1] for i in range(0, len(ids)-S, S)][:10]   # 10 non-overlapping (inp=tgt shifted)
print('num seqs', len(seqs), 'S', S)
lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.0
opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(b1,b2), eps=eps, weight_decay=wd)
losses=[]
for i,seq in enumerate(seqs):
    inp=seq[:-1].unsqueeze(0); tgt=seq[1:].unsqueeze(0)
    opt.zero_grad()
    logits=m(inp).logits
    loss=torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)), tgt.view(-1))
    loss.backward(); opt.step()
    losses.append(loss.item())
    print(f'step {i}: loss={loss.item():.6f}')
d='test/grad'
np.array(losses, dtype=np.float32).tofile(f'{d}/g3_losses.bin')
# save inputs/targets for the C side + final layer-0 q weight (sanity)
all_inp=np.concatenate([s[:-1].numpy() for s in seqs]).astype(np.int32); all_inp.tofile(f'{d}/g3_inputs.bin')
all_tgt=np.concatenate([s[1:].numpy()  for s in seqs]).astype(np.int32); all_tgt.tofile(f'{d}/g3_targets.bin')
# final weights to compare (layer 0 q, layer 15 down, lnfw)
pd={n:p for n,p in m.named_parameters()}
pd['model.layers.0.self_attn.q_proj.weight'].detach().numpy().astype(np.float32).tofile(f'{d}/g3_w_l0_q.bin')
pd['model.layers.15.mlp.down_proj.weight'].detach().numpy().astype(np.float32).tofile(f'{d}/g3_w_l15_d.bin')
pd['model.norm.weight'].detach().numpy().astype(np.float32).tofile(f'{d}/g3_w_lnfw.bin')
print('saved losses + inputs/targets + final weights')
