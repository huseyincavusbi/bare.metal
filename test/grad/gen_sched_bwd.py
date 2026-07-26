import torch, numpy as np, os
torch.manual_seed(8)
BT, C1, OC1, OC2, eps = 4, 16, 8, 5, 1e-5
inp  = torch.randn(BT, C1, requires_grad=True)
w1   = torch.randn(OC1, C1, requires_grad=True)   # [8,16]
rmsw = torch.randn(OC1, requires_grad=True)        # [8]
w2   = torch.randn(OC2, OC1, requires_grad=True)  # [5,8]
h = inp @ w1.T                                     # [4,8]
ss = (h*h).sum(-1,keepdim=True)/OC1
hn = h * torch.rsqrt(ss+eps) * rmsw                # [4,8]
out = hn @ w2.T                                    # [4,5]
gout = torch.randn(BT, OC2)
(out*gout).sum().backward()
d='test/grad'
for nm,t in [('sb_inp',inp),('sb_w1',w1),('sb_rmsw',rmsw),('sb_w2',w2),('sb_gout',gout),('sb_ginp',inp.grad),('sb_gw1',w1.grad),('sb_grmsw',rmsw.grad),('sb_gw2',w2.grad)]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin'); print(nm, tuple(t.shape))
print(f'BT={BT} C1={C1} OC1={OC1} OC2={OC2} eps={eps}')

# also dump forward intermediates for stage-by-stage checking
for nm,t in [('sb_h',h.detach()),('sb_hn',hn.detach()),('sb_out',out.detach())]:
    t.numpy().astype(np.float32).tofile(f'test/grad/{nm}.bin'); print('fwd', nm, tuple(t.shape))
