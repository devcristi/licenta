import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from pathlib import Path
import json

# Adăugăm calea către modulele BraTS
import sys
sys.path.append(str(Path(__file__).parent.parent / "BraTs2024"))
from model import get_brats_model  # type: ignore
from data_loader import train_transforms, val_transforms  # type: ignore

# CONFIGURARE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LUMIERE_CHECKPOINT = r"D:/study/licenta/creier/checkpoints/lumiere_finetune/best_lumiere_model.pth"
BRATS_PRETRAINED = r"D:/study/licenta/creier/checkpoints/best_model.pth"

# Alegem calea de resume: dacă există modelul LUMIERE, îl folosim, altfel cel BraTS
if os.path.exists(LUMIERE_CHECKPOINT):
    PRETRAINED_PATH = LUMIERE_CHECKPOINT
    IS_RESUME = True
else:
    PRETRAINED_PATH = BRATS_PRETRAINED
    IS_RESUME = False

LUMIERE_JSON = r"D:/study/licenta/creier/dataset/LUMIERE/lumiere_metadata.json"
CHECKPOINT_DIR = r"D:/study/licenta/creier/checkpoints/lumiere_finetune"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hiperparametri Fine-tuning
LR = 1e-5  # Rată de învățare și mai mică pentru rafinare
EPOCHS = 50 # Creștem numărul de epoci pentru că reluăm
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
FREEZE_ENCODER_EPOCHS = 0 # Oprim freeze-ul pentru că reluăm fine-tuning-ul complet

def get_brats_regions(y):
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0
    et = y[:, [3]] > 0
    return torch.cat([wt, tc, et], dim=1).float()

def fine_tune():
    # 1. Încărcare Date
    with open(LUMIERE_JSON, "r") as f:
        data = json.load(f)
    
    train_files = [d for d in data if d["split"] == "train"]
    val_files = [d for d in data if d["split"] == "val"]
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_ds = Dataset(data=val_files[:20], transform=val_transforms) # Validăm pe un subset pentru viteză
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. Model & Loss
    model = get_brats_model(in_channels=4, out_channels=4).to(DEVICE)
    
    if IS_RESUME:
        print(f"RELUĂM antrenarea din checkpoint-ul LUMIERE: {PRETRAINED_PATH}")
    else:
        print(f"Începem fine-tuning din modelul pre-antrenat BraTS: {PRETRAINED_PATH}")
    
    try:
        checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        # Dacă modelul a fost salvat cu DataParallel, scoatem prefixul 'module.'
        if all(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
        print("Greutățile au fost încărcate cu succes.")
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        print("Vom continua cu modelul neinițializat (nu este recomandat pentru fine-tuning).")
    
    # Ponderea CE: punem mai mult accent pe clasele tumorale (1, 2, 3) decât pe background (0)
    loss_function = DiceCELoss(
        to_onehot_y=True, softmax=True, 
        weight=torch.tensor([0.5, 2.0, 1.0, 2.0]).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()

    # Metrică
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") # include_background=True pentru că regiunile sunt deja filtrate
    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_dice = 0.7279 if IS_RESUME else 0.0

    for epoch in range(EPOCHS):
        model.train()
        
        # Logică de Freezing
        if epoch < FREEZE_ENCODER_EPOCHS:
            print(f"Epoch {epoch+1}: Encoder înghețat (primele {FREEZE_ENCODER_EPOCHS} epoci).")
            for name, param in model.named_parameters():
                if "model.0" in name: # Doar encoder-ul (downsampling)
                    param.requires_grad = False
        else:
            if epoch == FREEZE_ENCODER_EPOCHS:
                print(f"Epoch {epoch+1}: Dezghețăm tot modelul pentru fine-tuning complet.")
            for param in model.parameters():
                param.requires_grad = True

        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(DEVICE), batch_data["label"].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if step % 100 == 0:
                print(f"  Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss Mediu: {epoch_loss/len(train_loader):.4f}")

        # Validare
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(DEVICE), val_data["label"].to(DEVICE)
                # Folosim sliding window
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                
                val_outputs = [post_seg(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                # Convertim în regiuni BraTS pentru metrică (WT, TC, ET)
                # get_brats_regions returnează [1, 3, D, H, W], deci dăm squeeze(0) pentru a avea [3, D, H, W]
                val_outputs_reg = [get_brats_regions(i.unsqueeze(0)).squeeze(0) for i in val_outputs]
                val_labels_reg = [get_brats_regions(i.unsqueeze(0)).squeeze(0) for i in val_labels]
                
                dice_metric(y_pred=val_outputs_reg, y=val_labels_reg)
            
            metric_batch = dice_metric.aggregate()
            
            # metric_batch ar trebui să aibă acum dimensiunea [3] (WT, TC, ET)
            dice_wt = metric_batch[0].item()
            dice_tc = metric_batch[1].item()
            dice_et = metric_batch[2].item()
            mean_dice = torch.mean(metric_batch).item()
            
            dice_metric.reset()
            print(f"Validare - Mean Dice: {mean_dice:.4f}")
            print(f"  > WT (Whole Tumor): {dice_wt:.4f}")
            print(f"  > TC (Tumor Core):  {dice_tc:.4f}")
            print(f"  > ET (Enhancing):   {dice_et:.4f}")

            # Actualizăm scheduler-ul
            scheduler.step(mean_dice)
            
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_lumiere_model.pth"))
                print(f"Model nou salvat! (Dice: {best_dice:.4f})")


if __name__ == "__main__":
    fine_tune()
