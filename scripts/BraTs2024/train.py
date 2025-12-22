import os
import torch
from torch.cuda.amp import GradScaler, autocast
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from pathlib import Path

from data_loader import build_brats2024_list_from_json, train_transforms, val_transforms
from model import get_brats_model

# --- Configurație Optimizată pentru 6GB VRAM ---
CONFIG = {
    "json_path": "../../dataset/BRATS/brats_metadata_splits.json",
    "batch_size": 2,           
    "grad_accumulation": 2,    
    "epochs": 50,              
    "lr": 2e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_dir": "./checkpoints",
    "val_interval": 5,         # Validăm mai rar pentru a câștiga timp
    "max_steps_per_epoch": 250 # Limităm pașii per epocă pentru progres rapid
}

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

    # Folosim Dataset simplu (fără cache pe disc) pentru că nu avem spațiu (ne trebuie >100GB)
    # Dar mărim num_workers la 8 pentru că ai 12 nuclee CPU
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 2. Model, Loss, Optimizer
    model = get_brats_model(in_channels=4, out_channels=4).to(CONFIG["device"])
    
    # --- LOGICĂ RESUME ---
    checkpoint_path = os.path.join(CONFIG["model_dir"], "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
    
    loss_function = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # API nou pentru GradScaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Metrici separate pentru regiunile BraTS
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch") # mean_batch ne dă scor per clasă
    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    best_metric = -1
    
    print(f"Start Training! Device: {CONFIG['device']}")
    print(f"Sfat: Monitorizati temperatura laptopului. Daca sare de 85-90C, opriti-l.")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(CONFIG["device"]), batch_data["label"].to(CONFIG["device"])

            # API nou pentru autocast
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
                print(f"Ep {epoch+1} - Step {step}/{CONFIG['max_steps_per_epoch']} - Loss: {loss.item()*CONFIG['grad_accumulation']:.4f}")

            if step >= CONFIG["max_steps_per_epoch"]:
                break

        print(f"Epoch {epoch+1} Loss: {epoch_loss/step:.4f}")

        # 3. Validare periodică
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(CONFIG["device"]), val_data["label"].to(CONFIG["device"])
                    
                    # Optimizat: sw_batch_size=4 și overlap=0.25 pentru viteză
                    from monai.inferers import sliding_window_inference
                    val_outputs = sliding_window_inference(
                        val_inputs, (96, 96, 96), sw_batch_size=4, predictor=model, overlap=0.25
                    )
                    
                    val_outputs = [post_seg(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # Calculăm metricile pe clase (1=Necrotic, 2=Edema, 3=Enhancing)
                metric_batch = dice_metric.aggregate() 
                dice_metric.reset()
                
                # BraTS Regions:
                # WT = 1+2+3, TC = 1+3, ET = 3
                # Aici aproximăm prin media claselor pentru simplitate în log, 
                # dar afișăm scorul pentru fiecare componentă:
                d_necrotic = metric_batch[0].item()
                d_edema = metric_batch[1].item()
                d_enhancing = metric_batch[2].item()
                
                avg_dice = (d_necrotic + d_edema + d_enhancing) / 3
                
                if avg_dice > best_metric:
                    best_metric = avg_dice
                    torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], "best_model.pth"))
                    print(f"Model nou salvat! Mean Dice: {avg_dice:.4f}")
                
                log_msg = (
                    f"Epoca {epoch+1} - Mean Dice: {avg_dice:.4f} | "
                    f"Necrotic: {d_necrotic:.4f}, Edema: {d_edema:.4f}, Enhancing: {d_enhancing:.4f}\n"
                )
                with open(os.path.join(CONFIG["model_dir"], "train_log.txt"), "a") as f:
                    f.write(log_msg)

                print(f"Validare Ep {epoch+1}:")
                print(f" > Mean Dice: {avg_dice:.4f}")
                print(f" > Dice Necrotic (1): {d_necrotic:.4f}")
                print(f" > Dice Edema (2): {d_edema:.4f}")
                print(f" > Dice Enhancing (3): {d_enhancing:.4f}")

if __name__ == "__main__":
    train()
