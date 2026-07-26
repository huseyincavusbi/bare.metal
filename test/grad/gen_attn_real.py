import torch, numpy as np, os
torch.manual_seed(11)
NH, NKV, HD, S = 9, 3, 64, 4   # SmolLM2-135M sizes
kv_mul = NH // NKV
scale = 1.0/HD**0.5
Q = torch.randn(S, NH, HD, requires_grad=True)
K = torch.randn(S, NKV, HD, requires_grad=True)
V = torch.randn(S, NKV, HD, requires_grad=True)
out = torch.zeros(S, NH, HD)
for h in range(NH):
    kh = h // kv_mul
    for i in range(S):
        scores = (Q[i,h] @ K[:i+1,kh].T) * scale
        p = torch.softmax(scores, dim=-1)
        out[i,h] = p @ V[:i+1,kh]
gout = torch.randn(S, NH, HD)
(out*gout).sum().backward()
d='test/grad'
for nm,t in [('attn_real_Q',Q),('attn_real_K',K),('attn_real_V',V),('attn_real_gout',gout),('attn_real_gQ',Q.grad),('attn_real_gK',K.grad),('attn_real_gV',V.grad),('attn_real_out',out)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin')
print(f'NH={NH} NKV={NKV} HD={HD} S={S} scale={scale}')
