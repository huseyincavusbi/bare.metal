import torch, numpy as np, os
torch.manual_seed(2)
NH, NKV, HD, pos, theta = 6, 2, 8, 3, 10000.0
hd2 = HD // 2
q = torch.randn(NH, HD, requires_grad=True)
k = torch.randn(NKV, HD, requires_grad=True)
gout_q = torch.randn(NH, HD)
gout_k = torch.randn(NKV, HD)

def rotate(x, pos, theta, HD):
    hd2 = HD // 2
    out = x.clone()
    for i in range(hd2):
        freq = theta ** (-(2*i)/HD)
        c = torch.cos(torch.tensor(float(pos)*freq)); s = torch.sin(torch.tensor(float(pos)*freq))
        out[..., i]      = x[..., i]*c - x[..., i+hd2]*s
        out[..., i+hd2]  = x[..., i]*s + x[..., i+hd2]*c
    return out

q_rot = rotate(q, pos, theta, HD)
k_rot = rotate(k, pos, theta, HD)
loss = (q_rot * gout_q).sum() + (k_rot * gout_k).sum()
loss.backward()

d = 'test/grad'
def dump(name, t):
    a = t.detach().numpy().astype(np.float32); a.tofile(f'{d}/{name}.bin')
    print(f'{name}: {tuple(a.shape)}')
dump('rope_gout_q', gout_q); dump('rope_gout_k', gout_k)
dump('rope_gq', q.grad);     dump('rope_gk', k.grad)
print(f'NH={NH} NKV={NKV} HD={HD} pos={pos} theta={theta}')
