import torch
from model import get_brats_model
import os

def save_summary():
    model = get_brats_model()
    # Căi absolute pentru a evita erorile de folder
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    output_path = os.path.join(base_dir, "checkpoints", "model_summary.txt")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== STRUCTURA MODELULUI U-NET 3D ===\n\n")
        
        if os.path.exists(checkpoint_path):
            f.write(f"Status: Incarcat din {checkpoint_path}\n")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            f.write(f"Numar total de straturi (parametri): {len(state_dict)}\n\n")
        else:
            f.write("Status: Model nou (neantrenat)\n\n")
            
        f.write("--- DETALII STRATURI ---\n")
        for name, param in model.named_parameters():
            f.write(f"Layer: {name} | Shape: {list(param.shape)} | Trainable: {param.requires_grad}\n")
            
    print(f"Sumarul modelului a fost salvat in: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    save_summary()
