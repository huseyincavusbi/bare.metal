import torch, numpy as np, os
torch.manual_seed(9)
NH, NKV, HD, theta = 4, 2, 8, 10000.0
S = 5                          # sequence length; pos = position index 0..S-1
hd2 = HD // 2
kv_mul = NH // NKV
# [S, NH, HD] layout (matches matmul output + attention)
Q = torch.randn(S, NH, HD, requires_grad=True)
K = torch.randn(S, NKV, HD, requires_grad=True)
gout_q = torch.randn(S, NH, HD)
gout_k = torch.randn(S, NKV, HD)
def rotate_seq(x, theta, HD):
    # x: [S, H, HD]; pos = position index (row)
    hd2 = HD // 2
    out = x.clone()
    S_, H, _ = x.shape
    for s in range(S_):
        for i in range(hd2):
            freq = theta ** (-(2*i)/HD)
            c = torch.cos(torch.tensor(float(s)*freq)); s_ = torch.sin(torch.tensor(float(s)*freq))
            out[s,:,i]      = x[s,:,i]*c - x[s,:,i+hd2]*s_
            out[s,:,i+hd2]  = x[s,:,i]*s_ + x[s,:,i+hd2]*c
    return out
Q_rot = rotate_seq(Q, theta, HD)
K_rot = rotate_seq(K, theta, HD)
loss = (Q_rot*gout_q).sum() + (K_rot*gout_k).sum()
loss.backward()
d='test/grad'
for nm,t in [('roseq_Q',Q),('roseq_K',K),('roseq_gout_q',gout_q),('roseq_gout_k',gout_k),('roseq_gQ',Q.grad),('roseq_gK',K.grad)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape))
print(f'NH={NH} NKV={NKV} HD={HD} S={S} theta={theta} layout=[S,NH,HD]')
