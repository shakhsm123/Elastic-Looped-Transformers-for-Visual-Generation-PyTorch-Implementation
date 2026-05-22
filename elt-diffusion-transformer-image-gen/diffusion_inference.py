import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from vae import vae_decode

class NoiseScheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)

    def add_noise(self, x0, t, epsilon):
        t_cpu = t.cpu()
        sqrt_ab = self.sqrt_alpha_bars[t_cpu].reshape(-1, 1, 1, 1).to(x0.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t_cpu].reshape(-1, 1, 1, 1).to(x0.device)
        x_t = sqrt_ab * x0 + sqrt_omab * epsilon
        return x_t

    def remove_noise(self, x_t, t, predicted_epsilon):
        t_cpu = t if isinstance(t, int) else t.cpu()
        recip = self.sqrt_recip_alphas[t_cpu].to(x_t.device)
        beta = self.betas[t_cpu].to(x_t.device)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t_cpu].to(x_t.device)
        x_previous = recip * (x_t - beta / sqrt_omab * predicted_epsilon)
        if t_cpu > 0:
            x_previous += torch.sqrt(beta) * torch.randn_like(x_t)
        return x_previous

def ddpm_generate(model, noise_scheduler, dcfg, n_loops, device, vae):
  model.eval()
  x=torch.randn(1, 4, 32, 32).to(device)
  with torch.no_grad():
    timesteps = list(reversed(range(0, dcfg.T, dcfg.T // dcfg.sampling_steps)))
    for t in timesteps:
      t_batch=torch.tensor([t]).to(device)
      predicted_noise=model(x, t_batch, n_loops=n_loops)
      x=noise_scheduler.remove_noise(x, t, predicted_noise)
    image=vae_decode(vae, x)
  return image

def diffusion_any_time_inference(model, noise_scheduler, dcfg, device, vae):
  model.eval()
  generated_images=[]
  loop_counts=[1,2,3,4]
  for n_loops in loop_counts:
    image=ddpm_generate(model, noise_scheduler, dcfg, n_loops, device, vae)
    image=image.squeeze(0)
    image=image.permute(1,2,0)
    image=(image+1)/2
    image=image.cpu().numpy()
    image=np.clip(image, 0, 1)
    generated_images.append(image)
  fig, axes = plt.subplots(1, 4, figsize=(16, 4))
  fig.suptitle('Any-Time Inference for Diffusion — more loops = better quality', fontsize=14)

  for idx, (ax, img, n_loops) in enumerate(zip(axes, generated_images, loop_counts)):
      ax.imshow(img)
      ax.set_title(f'L={n_loops}')
      ax.axis('off')

  plt.tight_layout()
  plt.show()
  
