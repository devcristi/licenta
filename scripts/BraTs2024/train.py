import os
# Reduce noisy logs from TensorFlow/oneDNN and silence Python warnings during training
# Set before heavy imports that may trigger backend messages
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings
warnings.filterwarnings("ignore")
import random
import torch
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent, Compose
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from data_loader import build_brats2024_list_from_json, train_transforms, val_transforms
from model import get_brats_model

# --- Configurație SOTA ---
CONFIG = {
    "json_path": "../../dataset/BRATS/brats_metadata_splits.json",
    "batch_size": 4,           
    "grad_accumulation": 2,    
    "epochs": 200,             
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
    # Inițializează TensorBoard
    writer = SummaryWriter()
    
    # Verificare extra pentru CUDA
    print(f"Checking CUDA inside script: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found, running on CPU (will be very slow!)")

    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    set_determinism(seed=42)
    # Optimize cuDNN for fixed-size inputs (may improve throughput)
    torch.backends.cudnn.benchmark = True

    # 1. Data Loaders
    base_dir = Path(__file__).parent.parent.parent
    json_path = base_dir / "dataset" / "BRATS" / "brats_metadata_splits.json"
    
    train_files = build_brats2024_list_from_json(str(json_path), split="train")
    val_files = build_brats2024_list_from_json(str(json_path), split="val_from_train")
    
    # Amestecă înainte să tai - ca să nu iei doar primii 20 care poate sunt toți de la același spital
    random.seed(42)
    random.shuffle(val_files)
    
    # Subset pentru validare rapidă (20 volume)
    val_files_subset = val_files[:20]

    # Revenim la num_workers=4 pentru a elimina bottleneck-ul de CPU
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_ds = Dataset(data=val_files_subset, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=4,
        persistent_workers=True
    )

    # 2. Model, Loss, Optimizer
    model = get_brats_model(in_channels=4, out_channels=4).to(CONFIG["device"])
    
    # Selectare loss: păstrăm DiceFocalLoss implicit, dar permitem Focal Tversky
    use_tversky = os.environ.get('TVERSKY_LOSS', '0') == '1'

    if use_tversky:
        class FocalTverskyLoss(torch.nn.Module):
            def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-7):
                super().__init__()
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                self.eps = eps

            def forward(self, inputs, targets):
                probs = torch.softmax(inputs, dim=1)
                p = probs.view(probs.size(0), probs.size(1), -1)
                t = targets.view(targets.size(0), targets.size(1), -1)
                tp = (p * t).sum(-1)
                fp = (p * (1 - t)).sum(-1)
                fn = ((1 - p) * t).sum(-1)
                tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
                loss = (1 - tversky) ** self.gamma
                return loss.mean()

        loss_function = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        print("Using Focal Tversky Loss for training. Alpha: 0.3, Beta: 0.7, Gamma: 1.33")
        # directory for saving Tversky-run artifacts (never overwrite existing files)
        TVERSKY_DIR = os.path.join(CONFIG["model_dir"], "tversky_loss")
        os.makedirs(TVERSKY_DIR, exist_ok=True)
    else:
        # DiceFocalLoss: Focal loss focusează pe exemple dificile, fără ponderi manuale
        loss_function = DiceFocalLoss(
            smooth_nr=1e-5, smooth_dr=1e-5, 
            to_onehot_y=True, softmax=True,
            gamma=2.0,           # Factor focal pentru exemple dificile
            lambda_dice=1.0,     # Pondere Dice
            lambda_focal=1.0     # Pondere Focal
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    
    # Scheduler pentru a rafina invatarea
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # --- LOGICĂ RESUME COMPLETĂ ---
    start_epoch = 0
    best_metric = -1
    checkpoint_path = os.path.join(CONFIG["model_dir"], "latest_checkpoint.pth")

    # If running with TVERSKY_LOSS we want a fresh start (do not resume)
    if use_tversky:
        print("TVERSKY_LOSS=1 detected — starting training from scratch (ignoring existing checkpoints).")
    else:
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
    hd95_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)

    post_seg = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    
    # Post-procesare: Păstrăm doar componenta cea mai mare pentru ET (SOTA trick)
    post_processing_et = KeepLargestConnectedComponent(applied_labels=[3])

    print(f"Start Training SOTA! Device: {CONFIG['device']}")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        step = 0
        import time
        last_log_time = time.perf_counter()
        gpu_peak_mb = 0
        
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
                # Print current loss value and loss type (not Dice metric)
                loss_value = loss.item() * CONFIG["grad_accumulation"]
                loss_name = "Tversky" if use_tversky else "DiceFocal"
                print(f"Ep {epoch+1} - Step {step}/{CONFIG['max_steps_per_epoch']} - Loss ({loss_name}): {loss_value:.4f}")

            # Periodic throughput / memory logging (every 10 steps)
            if step % 10 == 0:
                now = time.perf_counter()
                elapsed = now - last_log_time
                last_log_time = now
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                    gpu_peak_mb = max(gpu_peak_mb, mem_reserved)
                    print(f"  [Perf] step_time={elapsed:.3f}s | mem_alloc={mem_alloc:.1f}MB | mem_reserved={mem_reserved:.1f}MB | peak_reserved={gpu_peak_mb:.1f}MB")
                else:
                    print(f"  [Perf] step_time={elapsed:.3f}s")

            if step >= CONFIG["max_steps_per_epoch"]:
                break

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_epoch_loss = epoch_loss / step
        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} | LR: {current_lr:.6f}")
        
        # Logare în TensorBoard
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch + 1)
        writer.add_scalar("learning_rate", current_lr, epoch + 1)

        # Forțăm eliberarea memoriei înainte de validare, care este intensiv
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Validare periodică cu TTA și Regiuni BraTS
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                run_hd95 = (epoch + 1) % 5 == 0
                # Initialize accumulators for sensitivity/specificity per-class (WT, TC, ET)
                sens_tp = [0, 0, 0]
                sens_fp = [0, 0, 0]
                sens_tn = [0, 0, 0]
                sens_fn = [0, 0, 0]

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
                        if run_hd95:
                            hd95_metric(y_pred=p, y=l)

                        # accumulate TP/FP/TN/FN per class for sensitivity/specificity
                        p_arr = p.squeeze(0)
                        l_arr = l.squeeze(0)
                        for ci in range(3):
                            pred_mask = p_arr[ci] > 0
                            label_mask = l_arr[ci] > 0
                            tp = int((pred_mask & label_mask).sum().item())
                            fp = int((pred_mask & (~label_mask)).sum().item())
                            fn = int(((~pred_mask) & label_mask).sum().item())
                            tn = int(((~pred_mask) & (~label_mask)).sum().item())
                            sens_tp[ci] += tp
                            sens_fp[ci] += fp
                            sens_fn[ci] += fn
                            sens_tn[ci] += tn

                metric_batch_dice = dice_metric.aggregate()
                dice_metric.reset()

                if run_hd95:
                    metric_batch_hd95 = hd95_metric.aggregate()
                    hd95_metric.reset()

                # compute sensitivity and specificity per class from accumulators
                metric_batch_sens_list = []
                metric_batch_spec_list = []
                for ci in range(3):
                    tp = sens_tp[ci]
                    fp = sens_fp[ci]
                    fn = sens_fn[ci]
                    tn = sens_tn[ci]
                    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                    metric_batch_sens_list.append(sens)
                    metric_batch_spec_list.append(spec)

                metric_batch_sens = torch.tensor(metric_batch_sens_list)
                metric_batch_spec = torch.tensor(metric_batch_spec_list)

                # Verificăm dimensiunea pentru a evita IndexError
                if metric_batch_dice.numel() >= 3:
                    d_wt, d_tc, d_et = metric_batch_dice[0].item(), metric_batch_dice[1].item(), metric_batch_dice[2].item()
                    s_wt, s_tc, s_et = metric_batch_sens[0].item(), metric_batch_sens[1].item(), metric_batch_sens[2].item()
                    sp_wt, sp_tc, sp_et = metric_batch_spec[0].item(), metric_batch_spec[1].item(), metric_batch_spec[2].item()
                    
                    if run_hd95 and metric_batch_hd95.numel() >= 3:
                        h_wt, h_tc, h_et = metric_batch_hd95[0].item(), metric_batch_hd95[1].item(), metric_batch_hd95[2].item()
                    else:
                        h_wt, h_tc, h_et = -1.0, -1.0, -1.0
                else:
                    # Fallback pentru cazul în care lipsește o clasă
                    d_wt = d_tc = d_et = metric_batch_dice.mean().item() if metric_batch_dice.numel() > 0 else 0
                    h_wt = h_tc = h_et = -1 # Valoare invalidă pentru HD95
                    s_wt = s_tc = s_et = metric_batch_sens.mean().item() if metric_batch_sens.numel() > 0 else 0
                    sp_wt = sp_tc = sp_et = metric_batch_spec.mean().item() if metric_batch_spec.numel() > 0 else 0

                
                avg_dice = (d_wt + d_tc + d_et) / 3

                # Logare în TensorBoard
                writer.add_scalar("Dice/validation_WT", d_wt, epoch + 1)
                writer.add_scalar("Dice/validation_TC", d_tc, epoch + 1)
                writer.add_scalar("Dice/validation_ET", d_et, epoch + 1)
                writer.add_scalar("Dice/validation_mean", avg_dice, epoch + 1)
                if run_hd95:
                    writer.add_scalar("HD95/validation_WT", h_wt, epoch + 1)
                    writer.add_scalar("HD95/validation_TC", h_tc, epoch + 1)
                    writer.add_scalar("HD95/validation_ET", h_et, epoch + 1)
                writer.add_scalar("Sensitivity/validation_WT", s_wt, epoch + 1)
                writer.add_scalar("Sensitivity/validation_TC", s_tc, epoch + 1)
                writer.add_scalar("Sensitivity/validation_ET", s_et, epoch + 1)
                writer.add_scalar("Specificity/validation_WT", sp_wt, epoch + 1)
                writer.add_scalar("Specificity/validation_TC", sp_tc, epoch + 1)
                writer.add_scalar("Specificity/validation_ET", sp_et, epoch + 1)

                if avg_dice > best_metric:
                    best_metric = avg_dice
                    torch.save(model.state_dict(), os.path.join(CONFIG["model_dir"], "best_model.pth"))
                    print(f"*** SOTA Model salvat! Mean Dice (WT/TC/ET): {avg_dice:.4f} ***)")
                    # If running with Tversky loss, also keep a timestamped copy in TVERSKY_DIR
                    if 'use_tversky' in locals() and use_tversky:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        bm_copy = os.path.join(TVERSKY_DIR, f"best_model_epoch{epoch+1}_{ts}.pth")
                        try:
                            torch.save(model.state_dict(), bm_copy)
                            with open(os.path.join(TVERSKY_DIR, f"best_model_epoch{epoch+1}_{ts}.txt"), 'w', encoding='utf-8') as mf:
                                mf.write(f"epoch: {epoch+1}\nmean_dice: {avg_dice:.6f}\nsaved_at: {ts}\n")
                        except Exception:
                            print(f"Warning: couldn't save best-model copy to {TVERSKY_DIR}")

                # Save per-epoch full checkpoint (model + optimizer + scheduler + metadata)
                epoch_checkpoint_path = os.path.join(CONFIG["model_dir"], f"checkpoint_epoch{epoch+1}.pth")
                epoch_model_path = os.path.join(CONFIG["model_dir"], f"model_epoch{epoch+1}.pth")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_metric': best_metric,
                    }, epoch_checkpoint_path)
                    # also keep a lightweight model-only copy for quick loading
                    torch.save(model.state_dict(), epoch_model_path)
                    # If Tversky mode, also save non-overwriting timestamped copies in TVERSKY_DIR
                    if 'use_tversky' in locals() and use_tversky:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        ckpt_copy = os.path.join(TVERSKY_DIR, f"checkpoint_epoch{epoch+1}_{ts}.pth")
                        model_copy = os.path.join(TVERSKY_DIR, f"model_epoch{epoch+1}_{ts}.pth")
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_metric': best_metric,
                            }, ckpt_copy)
                            torch.save(model.state_dict(), model_copy)
                            # also write a small metrics text file for quick scanning
                            with open(os.path.join(TVERSKY_DIR, f"metrics_epoch{epoch+1}_{ts}.txt"), 'w', encoding='utf-8') as mf:
                                mf.write(f"epoch: {epoch+1}\nmean_dice: {avg_dice:.6f}\nlr: {current_lr:.8f}\nsaved_at: {ts}\n")
                        except Exception:
                            print(f"Warning: couldn't save per-epoch copies to {TVERSKY_DIR}")
                except Exception as e:
                    # best-effort: don't break training if disk I/O fails
                    print(f"Warning: failed to save per-epoch checkpoint or model: {e}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                }, checkpoint_path)

                log_msg_base = (
                    f"Epoca {epoch+1} - Mean Dice: {avg_dice:.4f} | WT: {d_wt:.4f}, TC: {d_tc:.4f}, ET: {d_et:.4f} | "
                    f"Sens WT: {s_wt:.4f}, TC: {s_tc:.4f}, ET: {s_et:.4f} | "
                    f"Spec WT: {sp_wt:.4f}, TC: {sp_tc:.4f}, ET: {sp_et:.4f}"
                )
                if run_hd95:
                    log_msg_hd = f" | HD95 WT: {h_wt:.4f}, TC: {h_tc:.4f}, ET: {h_et:.4f}"
                    log_msg = log_msg_base + log_msg_hd + f" | LR: {current_lr:.6f}\n"
                else:
                    log_msg = log_msg_base + f" | LR: {current_lr:.6f}\n"

                with open(os.path.join(CONFIG["model_dir"], "train_log.txt"), "a") as f:
                    f.write(log_msg)

                print(f"Validare SOTA Ep {epoch+1}:")
                print(f"  > Dice (WT/TC/ET): {d_wt:.4f} / {d_tc:.4f} / {d_et:.4f}  (Mean: {avg_dice:.4f})")
                if run_hd95:
                    print(f"  > HD95 (WT/TC/ET): {h_wt:.4f} / {h_tc:.4f} / {h_et:.4f}")
                print(f"  > Sens (WT/TC/ET): {s_wt:.4f} / {s_tc:.4f} / {s_et:.4f}")
                print(f"  > Spec (WT/TC/ET): {sp_wt:.4f} / {sp_tc:.4f} / {sp_et:.4f}")

    
    writer.close()
    print("Antrenament finalizat și TensorBoard writer închis.")

if __name__ == "__main__":
    train()
