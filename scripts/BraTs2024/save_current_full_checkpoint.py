import torch
import os
from datetime import datetime

CKPT = r"d:\study\licenta\creier\checkpoints\latest_checkpoint.pth"
OUT_DIR = r"d:\study\licenta\creier\checkpoints"

ckpt = torch.load(CKPT, map_location='cpu')
if isinstance(ckpt, dict):
    epoch = ckpt.get('epoch', None)
    display_epoch = (epoch + 1) if epoch is not None else 'unknown'
    out_name = f"checkpoint_epoch{display_epoch}.pth"
    out_path = os.path.join(OUT_DIR, out_name)
    try:
        # Save the full checkpoint dict as-is
        torch.save(ckpt, out_path)
        meta_path = out_path + '.meta.txt'
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"source: {CKPT}\n")
            f.write(f"checkpoint_epoch: {epoch}\n")
            f.write(f"saved_at: {datetime.now()}\n")
        print('Saved full checkpoint to', out_path)
    except Exception as e:
        print('Failed to save full checkpoint:', e)
else:
    print('Checkpoint is not a dict; cannot save full checkpoint')
