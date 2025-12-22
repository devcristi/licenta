import torch
import torch.nn as nn
from monai.networks.nets import UNet

def get_brats_model(in_channels=4, out_channels=4):
    """
    Returneaza un model 3D U-Net optimizat pentru 6GB VRAM.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256), # Filtre reduse pentru memorie
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",                 # InstanceNorm e mai stabil pentru batch_size=1
        dropout=0.1,
    )
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_brats_model().to(device)
    dummy_input = torch.randn(1, 4, 96, 96, 96).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model creat pe: {device}")
    print(f"Output shape: {output.shape}")
