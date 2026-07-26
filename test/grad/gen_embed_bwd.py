import torch, numpy as np, os
torch.manual_seed(5)
# out[t] = wte[token[t]] ; backward: grad_wte[token[t]] += grad_out[t] (scatter-add)
# Use REPEATING tokens to exercise accumulation (atomic adds).
V, D, T = 8, 5, 6
tokens = torch.tensor([1, 3, 1, 0, 3, 2])   # tokens 1 and 3 repeat -> accumulate
wte = torch.randn(V, D, requires_grad=True)
out = wte[tokens]                            # (T, D) lookup
gout = torch.randn(T, D)
(out * gout).sum().backward()               # wte.grad[token] accumulates
d='test/grad'
for nm,t in [('embed_tokens',tokens),('embed_wte',wte),('embed_gout',gout),('embed_gwte',wte.grad)]:
    if t.dtype is torch.long:
        t.detach().numpy().astype(np.int32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape),'int32')
    else:
        t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape),'f32')
print(f'V={V} D={D} T={T}')
