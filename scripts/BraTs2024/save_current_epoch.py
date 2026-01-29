import torch
import os
from datetime import datetime

CKPT = r"d:\study\licenta\creier\checkpoints\latest_checkpoint.pth"
OUT_DIR = r"d:\study\licenta\creier\checkpoints"

ckpt = torch.load(CKPT, map_location='cpu')
if isinstance(ckpt, dict):
    epoch = ckpt.get('epoch', None)
    model_sd = None
    if 'model_state_dict' in ckpt:
        model_sd = ckpt['model_state_dict']
    elif all(hasattr(v, 'shape') for v in ckpt.values()):
        model_sd = ckpt

    if model_sd is None:
        print('No model state dict found in checkpoint.')
    else:
        # epoch in checkpoint is 0-based; display as epoch+1
        display_epoch = (epoch + 1) if epoch is not None else 'unknown'
        out_name = f"model_epoch{display_epoch}.pth"
        out_path = os.path.join(OUT_DIR, out_name)
        torch.save(model_sd, out_path)
        meta_path = out_path + '.meta.txt'
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"source: {CKPT}\n")
            f.write(f"checkpoint_epoch: {epoch}\n")
            f.write(f"saved_at: {datetime.now()}\n")
        print('Saved current checkpoint model_state_dict to', out_path)
else:
    print('Checkpoint is not a dict; cannot extract model_state_dict')
