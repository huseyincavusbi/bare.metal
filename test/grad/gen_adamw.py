import torch, numpy as np, os
torch.manual_seed(6)
N = 64
w0 = torch.randn(N)
g  = torch.randn(N)
param = torch.nn.Parameter(w0.clone())
param.grad = g.clone()
lr, (b1,b2), eps, wd = 1e-2, (0.9,0.999), 1e-8, 0.1
opt = torch.optim.AdamW([param], lr=lr, betas=(b1,b2), eps=eps, weight_decay=wd)
opt.step()  # t=1
st = opt.state[param]
d='test/grad'
for nm,t in [('adamw_w0',w0),('adamw_g',g),('adamw_w',param.detach()),('adamw_m',st['exp_avg']),('adamw_v',st['exp_avg_sq'])]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape))
print(f'lr={lr} b1={b1} b2={b2} eps={eps} wd={wd} t=1')
