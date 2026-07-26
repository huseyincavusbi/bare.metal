import torch, numpy as np, os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
torch.manual_seed(0)
md='data/smollm2-135m'
cfg=AutoConfig.from_pretrained(md); cfg.num_hidden_layers=2
tok=AutoTokenizer.from_pretrained(md)
m=AutoModelForCausalLM.from_pretrained(md, config=cfg, torch_dtype=torch.float32)
ids=torch.tensor([[6403, 1980, 253, 655]])
targets=torch.tensor([[1980, 253, 655, 0]])
logits=m(ids).logits
loss=torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
loss.backward()
d='test/grad'; os.makedirs(d, exist_ok=True)
pd={n:p for n,p in m.named_parameters()}
def dump(nm,name):
    g=pd[name].grad
    a=g.detach().numpy().astype(np.float32); a.tofile(f'{d}/{nm}.bin'); return a.size
D=m.config.hidden_size; NH=m.config.num_attention_heads; NKV=m.config.num_key_value_heads; HD=D//NH; H=m.config.intermediate_size
total=0
for l in [0,1]:
    total+=dump(f'b2_l{l}_qw',f'model.layers.{l}.self_attn.q_proj.weight')
    total+=dump(f'b2_l{l}_kw',f'model.layers.{l}.self_attn.k_proj.weight')
    total+=dump(f'b2_l{l}_vw',f'model.layers.{l}.self_attn.v_proj.weight')
    total+=dump(f'b2_l{l}_ow',f'model.layers.{l}.self_attn.o_proj.weight')
    total+=dump(f'b2_l{l}_ln1w',f'model.layers.{l}.input_layernorm.weight')
    total+=dump(f'b2_l{l}_ln2w',f'model.layers.{l}.post_attention_layernorm.weight')
    total+=dump(f'b2_l{l}_dw',f'model.layers.{l}.mlp.down_proj.weight')
total+=dump('b2_lnfw','model.norm.weight')
print(f'2-layer loss={loss.item():.4f} D={D} NH={NH} NKV={NKV} HD={HD} H={H}')
