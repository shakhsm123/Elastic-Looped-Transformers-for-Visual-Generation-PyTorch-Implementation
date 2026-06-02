import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.show = lambda: None

from diffusion_model import DiffusionELTConfig, ELT_DiT
from diffusion_inference import NoiseScheduler, ddpm_generate, diffusion_any_time_inference
from vae import load_vae
dcfg = DiffusionELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,
    L_max=4,
    batch_size=32,
)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_epoch_diffusion0.pt'

device=torch.device('cuda:0')

checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['model']
if next(iter(state_dict)).startswith("_orig_mod."):
    state_dict = {
        k.replace("_orig_mod.", "", 1): v
        for k, v in state_dict.items()
    }

print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

model = ELT_DiT(dcfg).to(device)
model.load_state_dict(state_dict)
model.eval()

vae = load_vae(device)
noise_scheduler = NoiseScheduler(dcfg)

diffusion_any_time_inference(model, noise_scheduler, dcfg, vae, device)
plt.savefig('generated_diffusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved to generated_diffusion.png")
