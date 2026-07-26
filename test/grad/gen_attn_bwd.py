import torch, numpy as np, os
torch.manual_seed(7)
NH, NKV, HD, S = 4, 2, 4, 3
kv_mul = NH // NKV
scale = 1.0/HD**0.5
# [S, NH, HD] layout -- matches matmul output [S, NH*HD] reshaped (no transpose)
Q = torch.randn(S, NH, HD, requires_grad=True)
K = torch.randn(S, NKV, HD, requires_grad=True)
V = torch.randn(S, NKV, HD, requires_grad=True)
out = torch.zeros(S, NH, HD)
for h in range(NH):
    kh = h // kv_mul
    for i in range(S):
        scores = (Q[i,h] @ K[:i+1,kh].T) * scale   # [i+1]
        p = torch.softmax(scores, dim=-1)
        out[i,h] = p @ V[:i+1,kh]                    # [HD]
gout = torch.randn(S, NH, HD)
(out * gout).sum().backward()
d='test/grad'
for nm,t in [('attn_Q',Q),('attn_K',K),('attn_V',V),('attn_gout',gout),('attn_gQ',Q.grad),('attn_gK',K.grad),('attn_gV',V.grad),('attn_out',out)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin')
print(f'NH={NH} NKV={NKV} HD={HD} S={S} kv_mul={kv_mul} scale={scale} layout=[S,NH,HD]')
