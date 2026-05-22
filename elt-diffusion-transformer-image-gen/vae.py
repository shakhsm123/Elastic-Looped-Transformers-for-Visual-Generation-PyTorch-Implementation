import torch
from diffusers import AutoencoderKL

def load_vae(device):
  vae_model=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
  vae_model.to(device)
  for parameter in vae_model.parameters():
    parameter.requires_grad=False
  vae_model.eval()
  return vae_model
def vae_encode(vae, images):
  with torch.no_grad():
    encoding=vae.encode(images).latent_dist.sample()
    encoding=encoding*0.18215
  return encoding
def vae_decode(vae, latents):
  with torch.no_grad():
    latents=latents/0.18215
    decoding=vae.decode(latents).sample
  return decoding
  
