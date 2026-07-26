import torch, numpy as np, os
torch.manual_seed(1)
N, C, eps = 4, 24, 1e-5
x = torch.randn(N, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
# RMSNorm forward (matches our kernel): out = x / sqrt(mean(x^2)+eps) * w
ss = (x*x).sum(-1, keepdim=True) / C
out = x * torch.rsqrt(ss + eps) * w
gout = torch.randn(N, C)
out.backward(gout)
d = 'test/grad'
os.makedirs(d, exist_ok=True)
def dump(name, t):
    a = t.detach().numpy().astype(np.float32); a.tofile(f'{d}/{name}.bin')
    print(f'{name}: {tuple(a.shape)}')
dump('rms_x', x); dump('rms_w', w); dump('rms_gout', gout)
dump('rms_gx', x.grad); dump('rms_gw', w.grad)
print(f'N={N} C={C} eps={eps}')
