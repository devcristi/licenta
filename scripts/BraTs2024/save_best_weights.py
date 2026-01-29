import torch
import os
from datetime import datetime

CKPT = r"d:\study\licenta\creier\checkpoints\latest_checkpoint.pth"
OUT_DIR = r"d:\study\licenta\creier\checkpoints"
OUT_NAME = f"best_extracted_from_latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

ckpt = torch.load(CKPT, map_location='cpu')
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model_sd = ckpt['model_state_dict']
    torch.save(model_sd, OUT_PATH)
    print('Saved model_state_dict to', OUT_PATH)
    # Also save a small metadata file
    meta_path = OUT_PATH + '.meta.txt'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"source: {CKPT}\n")
        f.write(f"epoch: {ckpt.get('epoch')}\n")
        f.write(f"best_metric: {ckpt.get('best_metric')}\n")
        f.write(f"saved_at: {datetime.now()}\n")
    print('Saved metadata to', meta_path)
else:
    print('Checkpoint does not contain model_state_dict; saving full checkpoint instead.')
    torch.save(ckpt, OUT_PATH)
    print('Saved full checkpoint to', OUT_PATH)
