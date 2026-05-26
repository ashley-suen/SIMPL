import torch, sys
sys.stdout.reconfigure(encoding='utf-8')
ckpt = torch.load('../saved_models/20260501-022610_llm_simpl_best.tar', map_location='cpu')
print('Top-level keys:', list(ckpt.keys()))
for k, v in ckpt.items():
    if isinstance(v, dict):
        print(f'  {k}: dict with {len(v)} entries')
    elif isinstance(v, torch.Tensor):
        print(f'  {k}: Tensor {v.shape}')
    else:
        print(f'  {k}: {type(v).__name__} = {v}')
