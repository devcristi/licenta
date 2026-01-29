import torch
import os
from datetime import datetime

CKPT = r"d:\study\licenta\creier\checkpoints\latest_checkpoint.pth"
print('File:', CKPT)
try:
    st = os.stat(CKPT)
    print('Size (bytes):', st.st_size)
    print('Last modified:', datetime.fromtimestamp(st.st_mtime))
except Exception as e:
    print('Cannot stat file:', e)

try:
    ckpt = torch.load(CKPT, map_location='cpu')
    print('Loaded checkpoint type:', type(ckpt))
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print('Top-level keys:', keys)
        # common keys
        for k in ('epoch', 'best_metric', 'best_score', 'metric', 'state_dict', 'model_state_dict', 'optimizer_state_dict'):
            if k in ckpt:
                print(f"{k}:", ckpt[k] if not hasattr(ckpt[k], 'keys') else f"dict(len={len(ckpt[k])})")
        # inspect state dict
        sd_key = None
        if 'state_dict' in ckpt:
            sd_key = 'state_dict'
        elif 'model_state_dict' in ckpt:
            sd_key = 'model_state_dict'
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd_key = None
        if sd_key:
            sd = ckpt[sd_key]
            print(f"Found state dict under '{sd_key}' with {len(sd)} items. Sample keys/shapes:")
            for i, (k, v) in enumerate(sd.items()):
                if hasattr(v, 'shape'):
                    print(f"  {i:03d}: {k} -> shape {tuple(v.shape)}")
                else:
                    print(f"  {i:03d}: {k} -> type {type(v)}")
                if i >= 19:
                    break
        else:
            # maybe ckpt itself is a state dict
            if isinstance(ckpt, dict) and all(hasattr(v, 'shape') for v in ckpt.values()):
                print('Checkpoint appears to be a raw state_dict with', len(ckpt), 'entries. Sample:')
                for i, (k, v) in enumerate(ckpt.items()):
                    print(f"  {i:03d}: {k} -> shape {tuple(v.shape)}")
                    if i >= 19:
                        break
    else:
        print('Checkpoint is not a dict; repr:')
        print(repr(ckpt)[:1000])
except Exception as e:
    print('Error loading checkpoint:', e)
