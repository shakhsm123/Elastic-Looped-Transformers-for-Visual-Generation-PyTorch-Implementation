import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

from diffusion_model import DiffusionELTConfig, ELT_DiT
from diffusion_inference import NoiseScheduler, ddpm_generate, diffusion_any_time_inference
from vae import load_vae

device = torch.device('cuda:0')

dcfg = DiffusionELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,
    L_max=4,
    batch_size=32,
)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_epoch_diffusion0.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')
print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

model = ELT_DiT(dcfg).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

vae = load_vae(device)
noise_scheduler = NoiseScheduler(dcfg)

diffusion_any_time_inference(model, noise_scheduler, dcfg, vae, device)
plt.savefig('generated_diffusion.png', dpi=150, bbox_inches='tight')
print("saved to generated_diffusion.png")
