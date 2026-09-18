import torch, numpy as np, os
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
md = 'data/smollm2-135m'
m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float32).train()
cfg = m.config
S = 8
inputs = np.fromfile('test/grad/g3_inputs.bin', dtype=np.int32)
targets = np.fromfile('test/grad/g3_targets.bin', dtype=np.int32)
ids = torch.tensor(inputs[:S].tolist()).long().unsqueeze(0)
tgt = torch.tensor(targets[:S].tolist()).long()
logits = m(ids).logits
loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt)
loss.backward()

pd = {n: p for n, p in m.named_parameters()}

def dump(nm, t):
    if t.grad is None:
        print('NO GRAD', nm); return
    t.grad.detach().numpy().astype(np.float32).tofile(f'test/grad/pt_{nm}.bin')

names = []
for l in range(cfg.num_hidden_layers):
    pref = f'model.layers.{l}.'
    for n, p in [
        ('input_layernorm.weight', 'ln1w'),
        ('self_attn.q_proj.weight', 'qw'),
        ('self_attn.k_proj.weight', 'kw'),
        ('self_attn.v_proj.weight', 'vw'),
        ('self_attn.o_proj.weight', 'ow'),
        ('post_attention_layernorm.weight', 'ln2w'),
        ('mlp.gate_proj.weight', 'gw'),
        ('mlp.up_proj.weight', 'uw'),
        ('mlp.down_proj.weight', 'dw'),
    ]:
        name = pref + n
        dump(f'g_{len(names):04d}', pd[name])
        names.append(name)
dump('g_%04d' % len(names), pd['model.norm.weight']); names.append('model.norm.weight')
dump('g_%04d' % len(names), pd['model.embed_tokens.weight']); names.append('model.embed_tokens.weight')

print(f'loss={loss.item():.4f} dumped {len(names)} grads')

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
opt.step()

ids1 = torch.tensor(inputs[S:2*S].tolist()).long().unsqueeze(0)
tgt1 = torch.tensor(targets[S:2*S].tolist()).long()
opt.zero_grad()
logits1 = m(ids1).logits
loss1 = torch.nn.functional.cross_entropy(logits1.view(-1, logits1.size(-1)), tgt1)
loss1.backward()

names1 = []
for l in range(cfg.num_hidden_layers):
    pref = f'model.layers.{l}.'
    for n, p in [
        ('input_layernorm.weight', 'ln1w'),
        ('self_attn.q_proj.weight', 'qw'),
        ('self_attn.k_proj.weight', 'kw'),
        ('self_attn.v_proj.weight', 'vw'),
        ('self_attn.o_proj.weight', 'ow'),
        ('post_attention_layernorm.weight', 'ln2w'),
        ('mlp.gate_proj.weight', 'gw'),
        ('mlp.up_proj.weight', 'uw'),
        ('mlp.down_proj.weight', 'dw'),
    ]:
        name = pref + n
        dump(f'g1_{len(names1):04d}', pd[name])
        names1.append(name)
dump('g1_%04d' % len(names1), pd['model.norm.weight']); names1.append('model.norm.weight')
dump('g1_%04d' % len(names1), pd['model.embed_tokens.weight']); names1.append('model.embed_tokens.weight')

print(f'step1 loss={loss1.item():.4f} dumped {len(names1)} step-1 grads')
