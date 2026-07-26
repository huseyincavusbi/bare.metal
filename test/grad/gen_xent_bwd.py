import torch, numpy as np, os
torch.manual_seed(4)
N, V = 3, 10
logits = torch.randn(N, V, requires_grad=True)
targets = torch.tensor([2, 7, 0])
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()
d='test/grad'
logits.detach().numpy().astype(np.float32).tofile(f'{d}/xent_logits.bin')
logits.grad.detach().numpy().astype(np.float32).tofile(f'{d}/xent_glogits.bin')
targets.detach().numpy().astype(np.int32).tofile(f'{d}/xent_targets.bin')  # int32!
print(f'N={N} V={V} loss={loss.item():.4f}  targets dumped as int32')
