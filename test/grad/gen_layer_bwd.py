import torch, numpy as np, os
torch.manual_seed(42)

S, HD, NH, NKV = 8, 16, 4, 1  # Sequence=8, HeadDim=16, NH=4, NKV=1 => D=64
D = NH * HD
KV = NKV * HD
eps = 1e-5
theta = 10000.0

inp = torch.randn(S, D, requires_grad=True)

# 1. Norm 1
rmsw1 = torch.randn(D, requires_grad=True)
def norm(x, w):
    ss = (x*x).mean(-1, keepdim=True)
    return x * torch.rsqrt(ss + eps) * w
n1 = norm(inp, rmsw1)

# 2. Q/K/V projection (using linear without bias)
wq = torch.randn(D, D, requires_grad=True)
wk = torch.randn(KV, D, requires_grad=True)
wv = torch.randn(KV, D, requires_grad=True)
q = n1 @ wq.T
k = n1 @ wk.T
v = n1 @ wv.T

# 3. RoPE
q_r = q.view(S, NH, HD)
k_r = k.view(S, NKV, HD)

def apply_rope(x, n_heads):
    out = torch.zeros_like(x)
    for s in range(S):
        for h in range(n_heads):
            for i in range(HD // 2):
                freq = 1.0 / (theta ** (2.0 * i / HD))
                c = np.cos(s * freq)
                s_ = np.sin(s * freq)
                x0 = x[s, h, i]
                x1 = x[s, h, i + HD // 2]
                out[s, h, i] = x0 * c - x1 * s_
                out[s, h, i + HD // 2] = x0 * s_ + x1 * c
    return out

q_rope = apply_rope(q_r, NH).view(S, D)
k_rope = apply_rope(k_r, NKV).view(S, KV)

# 4. Attention
q_a = q_rope.view(S, NH, HD).transpose(0, 1) # [NH, S, HD]
kv_mul = NH // NKV
k_a = k_rope.view(S, NKV, HD).transpose(0, 1).repeat_interleave(kv_mul, dim=0) # [NH, S, HD]
v_a = v.view(S, NKV, HD).transpose(0, 1).repeat_interleave(kv_mul, dim=0)

scores = (q_a @ k_a.transpose(-2, -1)) / np.sqrt(HD) # [NH, S, S]
mask = torch.tril(torch.ones(S, S))
scores.masked_fill_(mask == 0, float('-inf'))
probs = torch.softmax(scores, dim=-1)
attn_out = probs @ v_a # [NH, S, HD]
attn_out = attn_out.transpose(0, 1).reshape(S, D) # [S, D]

# 5. Projection
w_proj = torch.randn(D, D, requires_grad=True)
proj_out = attn_out @ w_proj.T

# 6. ADD 1
res1 = inp + proj_out

# 7. Norm 2
rmsw2 = torch.randn(D, requires_grad=True)
n2 = norm(res1, rmsw2)

# 8. FFN (SwiGLU)
HID = 128
w_gate = torch.randn(HID, D, requires_grad=True)
w_up = torch.randn(HID, D, requires_grad=True)
w_down = torch.randn(D, HID, requires_grad=True)

gate = n2 @ w_gate.T
up = n2 @ w_up.T
silu = gate * torch.sigmoid(gate)
ffn_out = silu * up
down_out = ffn_out @ w_down.T

# 9. ADD 2
res2 = res1 + down_out

# Backward
gout = torch.randn(S, D)
(res2 * gout).sum().backward()

# Dump
d = 'test/grad'
for nm, t in [
    ('layer_inp', inp), ('layer_rmsw1', rmsw1), ('layer_wq', wq), ('layer_wk', wk), ('layer_wv', wv),
    ('layer_wproj', w_proj), ('layer_rmsw2', rmsw2), ('layer_wgate', w_gate), ('layer_wup', w_up),
    ('layer_wdown', w_down), ('layer_gout', gout),
    ('layer_ginp', inp.grad), ('layer_grmsw1', rmsw1.grad), ('layer_gwq', wq.grad),
    ('layer_gwk', wk.grad), ('layer_gwv', wv.grad), ('layer_gwproj', w_proj.grad),
    ('layer_grmsw2', rmsw2.grad), ('layer_gwgate', w_gate.grad), ('layer_gwup', w_up.grad),
    ('layer_gwdown', w_down.grad),
    # forward checks
    ('layer_n1', n1), ('layer_q', q_rope), ('layer_k', k_rope), ('layer_attn_out', attn_out),
    ('layer_res1', res1), ('layer_n2', n2), ('layer_res2', res2)
]:
    t.detach().numpy().astype(np.float32).tofile(f'{d}/{nm}.bin')

print(f'S={S} D={D} NH={NH} NKV={NKV} HD={HD} HID={HID}')
