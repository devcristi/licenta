import os
import torch
from torch.cuda.amp import GradScaler, autocast
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose
from monai.utils import set_determinism
from pathlib import Path

from data_loader import build_brats2024_list_from_json, train_transforms, val_transforms
from model import get_brats_model

# --- Configurație SOTA ---
CONFIG = {
    "json_path": "../../dataset/BRATS/brats_metadata_splits.json",
    "batch_size": 2,           
    "grad_accumulation": 2,    
    "epochs": 100,             
    "lr": 2e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_dir": "D:/study/licenta/creier/checkpoints", # Cale absolută
    "val_interval": 1,         # Validare la fiecare epocă pentru feedback rapid
    "max_steps_per_epoch": 250 
}

def get_brats_regions(y):
    """
    Convertește cele 3 canale de tumoră în regiunile standard BraTS:
    WT (Whole Tumor): 1, 2, 3
    TC (Tumor Core): 1, 3
    ET (Enhancing Tumor): 3
    """ 
    # y are shape [B, 4, D, H, W] (one-hot)
    # Canal 0: BG, 1: Necrotic, 2: Edema, 3: Enhancing
    wt = torch.sum(y[:, 1:], dim=1, keepdim=True) > 0
    tc = torch.sum(y[:, [1, 3]], dim=1, keepdim=True) > 0
    et = y[:, [3]] > 0
    return torch.cat([wt, tc, et], dim=1).float()

def train():
    # Verificare extra pentru CUDA
    print(f"Checking CUDA inside script: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found, running on CPU (will be very slow!)")

    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    set_determinism(seed=42)

    # 1. Data Loaders
    base_dir = Path(__file__).parent.parent.parent
    json_path = base_dir / "dataset" / "BRATS" / "brats_metadata_splits.json"
    
    train_files = build_brats2024_list_from_json(str(json_path), split="train")
    val_files = build_brats2024_list_from_json(str(json_path), split="val_from_train")
    
    # Subset pentru validare rapidă (20 volume)
    val_files_subset = val_files[:20]

    # Revenim la num_workers=4 pentru a elimina bottleneck-ul de CPU
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_ds = Dataset(data=val_files_subset, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=2,
        persistent_workers=True
    )

    # 2. Model, Loss, Optimizer
    model = get_brats_model(in_channels=4, out_channels=4).to(CONFIG["device"])
    
    # Loss SOTA: Dice + CrossEntropy cu greutăți pentru clasele mici
    # Greutăți: BG=1, Necrotic=3, Edema=1, Enhancing=3
    class_weights = torch.tensor([1.0, 3.0, 1.0, 3.0]).to(CONFIG["device"])
    loss_function = DiceCELoss(
        smooth_nr=1e-5, smooth_dr=1e-5, 
        to_onehot_y=True, softmax=True,
        weight=class_weights
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # Scheduler pentru a rafina invatarea
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # --- LOGICĂ RESUME COMPLETĂ ---
    start_epoch = 0
    best_metric = -1
    checkpoint_path = os.path.join(CONFIG["model_dir"], "latest_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Reluăm antrenarea de la checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        print(f"Continuăm de la Epoca {start_epoch} (Best Dice: {best_metric:.4f})")
    elif os.path.exists(os.path.join(CONFIG["model_dir"], "best_model.pth")):
        print("Incarcam greutatile din best_model.pth")
        model.load_state_dict(torch.load(os.path.join(CONFIG["model_dir"], "best_model.pth"), map_location=CONFIG["device"]))
        # Încercăm să detectăm epoca din log dacă nu avem checkpoint
        log_path = os.path.join(CONFIG["model_dir"], "train_log.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    if "Epoca" in last_line:
                        try:
                            start_epoch = int(last_line.split(" ")[1])
                            print(f"Detectat din log: Continuăm de la Epoca {start_epoch}")
                        except: pass
    
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Metrici SOTA: WT, TC, ET
    # include_background=True pentru că get_brats_regions returnează deja doar regiunile de interes
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    
    # Post-procesare: Păstrăm doar componenta cea mai mare pentru ET (SOTA trick)
    post_processing_et = KeepLargestConnectedComponent(applied_labels=[3])

    print(f"Start Training SOTA! Device: {CONFIG['device']}")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(CONFIG["device"]), batch_data["label"].to(CONFIG["device"])

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss = loss / CONFIG["grad_accumulation"]

            if scaler:
                scaler.scale(loss).backward()
                if step % CONFIG["grad_accumulation"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if step % CONFIG["grad_accumulation"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * CONFIG["grad_accumulation"]
            
            if step % 50 == 0:
                # Explicăm că Loss = Dice + CE (de aia e mai mare acum)
                print(f"Ep {epoch+1} - Step {step}/{CONFIG['max_steps_per_epoch']} - Loss: {loss.item()*CONFIG['grad_accumulation']:.4f} (Dice+CE)")

            if step >= CONFIG["max_steps_per_epoch"]:
                break

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} Loss: {epoch_loss/step:.4f} | LR: {current_lr:.6f}")

        # 3. Validare periodică cu TTA și Regiuni BraTS
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(CONFIG["device"]), val_data["label"].to(CONFIG["device"])
                    
                    # --- TEST-TIME AUGMENTATION (TTA) ---
                    # Facem media între imaginea originală și flip-uri pe cele 3 axe
                    from monai.inferers import sliding_window_inference
                    
                    def get_pred(x):
                        return sliding_window_inference(x, (96, 96, 96), sw_batch_size=4, predictor=model, overlap=0.25)

                    val_outputs = get_pred(val_inputs)
                    # Flip X
                    val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[2])), dims=[2])
                    # Flip Y
                    val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[3])), dims=[3])
                    # Flip Z
                    val_outputs += torch.flip(get_pred(torch.flip(val_inputs, dims=[4])), dims=[4])
                    
                    val_outputs /= 4.0 # Media celor 4 predicții
                    
                    # Post-procesare și conversie în regiuni
                    val_outputs = [post_seg(i) for i in decollate_batch(val_outputs)]
                    # Aplicăm KeepLargestConnectedComponent pe predicție
                    val_outputs = [post_processing_et(i) for i in val_outputs]
                    
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    
                    # Convertim în regiuni WT, TC, ET pentru metrică
                    val_outputs_regions = [get_brats_regions(i.unsqueeze(0)) for i in val_outputs]
                    val_labels_regions = [get_brats_regions(i.unsqueeze(0)) for i in val_labels]
                    
                    for p, l in zip(val_outputs_regions, val_labels_regions):
                        dice_metric(y_pred=p, y=l)

                metric_batch = dice_metric.aggregate() 
                dice_metric.reset()
                
                # Verificăm dimensiunea pentru a evita IndexError
                if metric_batch.numel() >= 3:
                    d_wt = metric_batch[0].item()
                    d_tc = metric_batch[1].item()
                    d_et = metric_batch[2].item()
                else:
                    print(f"Warning: Metric batch size is {metric_batch.numel()}, expected 3.")
                    d_wt = d_tc = d_et = metric_batch.mean().item() if metric_batch.numel() > 0 else 0
                
                avg_dice = (d_wt + d_tc + d_et) / 3
                
                if avg_dice > best_metric:
                    best_metric = avg_dice
                    torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], "best_model.pth"))
                    print(f"*** SOTA Model salvat! Mean Dice (WT/TC/ET): {avg_dice:.4f} ***")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                }, checkpoint_path)

                log_msg = (
                    f"Epoca {epoch+1} - Mean WT/TC/ET: {avg_dice:.4f} | "
                    f"WT: {d_wt:.4f}, TC: {d_tc:.4f}, ET: {d_et:.4f} | LR: {current_lr:.6f}\n"
                )
                with open(os.path.join(CONFIG["model_dir"], "train_log.txt"), "a") as f:
                    f.write(log_msg)

                print(f"Validare SOTA Ep {epoch+1}:")
                print(f" > Whole Tumor (WT): {d_wt:.4f}")
                print(f" > Tumor Core (TC): {d_tc:.4f}")
                print(f" > Enhancing Tumor (ET): {d_et:.4f}")

if __name__ == "__main__":
    train()
