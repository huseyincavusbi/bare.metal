import torch, numpy as np, os
torch.manual_seed(12)
NH, NKV, HD, S, theta = 9, 3, 64, 4, 100000.0
hd2 = HD // 2
Q = torch.randn(S, NH, HD, requires_grad=True)
K = torch.randn(S, NKV, HD, requires_grad=True)
gout_q = torch.randn(S, NH, HD)
gout_k = torch.randn(S, NKV, HD)
out = torch.zeros_like(Q); outk = torch.zeros_like(K)
for s in range(S):
    for h in range(NH):
        for i in range(hd2):
            f = theta ** (-(2*i)/HD); c=torch.cos(torch.tensor(float(s)*f)); sn=torch.sin(torch.tensor(float(s)*f))
            out[s,h,i]=Q[s,h,i]*c-Q[s,h,i+hd2]*sn; out[s,h,i+hd2]=Q[s,h,i]*sn+Q[s,h,i+hd2]*c
    for h in range(NKV):
        for i in range(hd2):
            f = theta ** (-(2*i)/HD); c=torch.cos(torch.tensor(float(s)*f)); sn=torch.sin(torch.tensor(float(s)*f))
            outk[s,h,i]=K[s,h,i]*c-K[s,h,i+hd2]*sn; outk[s,h,i+hd2]=K[s,h,i]*sn+K[s,h,i+hd2]*c
((out*gout_q).sum()+(outk*gout_k).sum()).backward()
d='test/grad'
for nm,t in [('roreal_Q',Q),('roreal_K',K),('roreal_gout_q',gout_q),('roreal_gout_k',gout_k),('roreal_gQ',Q.grad),('roreal_gK',K.grad)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin')
print(f'NH={NH} NKV={NKV} HD={HD} S={S} theta={theta}')
