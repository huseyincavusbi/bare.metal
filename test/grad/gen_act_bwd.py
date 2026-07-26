import torch, numpy as np, os
torch.manual_seed(3)
N = 32
d = 'test/grad'; os.makedirs(d, exist_ok=True)

# ---- GELU (tanh approx, matches our kernel) ----
x = torch.randn(N, requires_grad=True)
cdf = 0.5*(1 + torch.tanh(0.79788456*x*(1+0.044715*x*x)))
out = x*cdf
gout = torch.randn(N)
(out*gout).sum().backward()
for nm,t in [('gelu_x',x),('gelu_gout',gout),('gelu_gx',x.grad)]:
    a=t.detach().numpy().astype(np.float32); a.tofile(f'{d}/{nm}.bin'); print(nm, tuple(a.shape))

# ---- SwiGLU: out = silu(gate)*up, silu(x)=x/(1+e^-x) ----
gate = torch.randn(N, requires_grad=True)
up   = torch.randn(N, requires_grad=True)
silu = gate/(1+torch.exp(-gate))
out2 = silu*up
gout2 = torch.randn(N)
(out2*gout2).sum().backward()
for nm,t in [('swiglu_gate',gate),('swiglu_up',up),('swiglu_gout',gout2),('swiglu_ggate',gate.grad),('swiglu_gup',up.grad)]:
    a=t.detach().numpy().astype(np.float32); a.tofile(f'{d}/{nm}.bin'); print(nm, tuple(a.shape))
print(f'N={N}')
