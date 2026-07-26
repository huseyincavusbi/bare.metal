import torch, numpy as np, os
torch.manual_seed(7)
NH, NKV, HD, S = 4, 2, 4, 3
kv_mul = NH // NKV
scale = 1.0/HD**0.5
Q = torch.randn(NH, S, HD, requires_grad=True)
K = torch.randn(NKV, S, HD, requires_grad=True)
V = torch.randn(NKV, S, HD, requires_grad=True)
# manual causal GQA forward (matches our kernel convention exactly)
out = torch.zeros(NH, S, HD)
for h in range(NH):
    kh = h // kv_mul
    for i in range(S):
        scores = (Q[h,i] @ K[kh,:i+1].T) * scale          # [i+1]
        p = torch.softmax(scores, dim=-1)
        out[h,i] = p @ V[kh,:i+1]                          # [HD]
gout = torch.randn(NH, S, HD)
(out * gout).sum().backward()
d='test/grad'
for nm,t in [('attn_Q',Q),('attn_K',K),('attn_V',V),('attn_gout',gout),('attn_gQ',Q.grad),('attn_gK',K.grad),('attn_gV',V.grad)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape))
print(f'NH={NH} NKV={NKV} HD={HD} S={S} kv_mul={kv_mul} scale={scale}')
