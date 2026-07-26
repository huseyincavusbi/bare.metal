import torch, numpy as np, os
# Reference data for matmul_backward validation.
# Forward: out[bt,oc] = bias[oc] + sum_i inp[bt,i] * w[oc,i]
# We compute PyTorch autograd grads and dump everything for the C/Metal side.
torch.manual_seed(0)
BT, C, OC = 3, 16, 8            # small, deterministic, easy to eyeball
inp = torch.randn(BT, C, requires_grad=True)
w   = torch.randn(OC, C, requires_grad=True)
bias= torch.randn(OC, requires_grad=True)
out = inp @ w.t() + bias         # (BT, OC)
gout = torch.randn(BT, OC)       # upstream grad (random)
out.backward(gout)

d = 'test/grad'
os.makedirs(d, exist_ok=True)
def dump(name, t):
    a = t.detach().numpy().astype(np.float32)
    a.tofile(f'{d}/{name}.bin')
    print(f'{name}: shape {a.shape} -> {d}/{name}.bin')

dump('inp', inp)
dump('w', w)
dump('bias', bias)
dump('gout', gout)             # grad_out (upstream)
dump('ginp', inp.grad)         # expected grad_input
dump('gw', w.grad)             # expected grad_weight
print(f'BT={BT} C={C} OC={OC}')
