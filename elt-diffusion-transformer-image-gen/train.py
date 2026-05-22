import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import datetime

from diffusion_model import DiffusionELTConfig, PatchEmbedding, LearnedPositionalEmbedding_Diffusion, TimeStepEmbedding, TransformerBlock, ELT_DiT
from diffusion_inference import NoiseScheduler

from diffusion_losses import diff_ILSD_train_step
from diffusion_dataset import ImageNet100Dataset, LatentDataset, CachedLatentDataset
from vae import load_vae

def main():
  LOCAL_RANK=int(os.environ.get("LOCAL_RANK"))
  RANK=int(os.environ.get("RANK"))
  WORLD_SIZE=int(os.environ.get("WORLD_SIZE"))

  dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))

  torch.cuda.set_device(LOCAL_RANK)
  device=torch.device(f'cuda:{LOCAL_RANK}')

  dcfg=DiffusionELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,
    L_max=4,
    batch_size=32,
  )
    
  latent_cache = "/home/gpuhead-1/datasets/imagenet100_latents_train.pt"

  if RANK == 0:
      if not os.path.exists(latent_cache):
          print("rank 0: encoding dataset...")
          vae = load_vae(device)
        
          train_transforms = transforms.Compose([
              transforms.Resize((256, 256)),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          ])
          train_dataset = ImageNet100Dataset(
              "/media/gpuhead-1/gpu-head-1-2nd/imagenet100/data",
              transform=train_transforms
          )
          latent_dataset = LatentDataset(train_dataset, vae, device)
          torch.save({
              'latents': latent_dataset.latents,
              'labels': latent_dataset.labels
          }, latent_cache)
          print("rank 0: cache saved")

  dist.barrier()
  cache = torch.load(latent_cache, map_location='cpu')
  train_token_dataset = CachedLatentDataset(cache['latents'], cache['labels'])
  train_sampler = DistributedSampler(
      train_token_dataset,
      num_replicas=WORLD_SIZE,
      rank=RANK,
      shuffle=True
  )
  train_loader = DataLoader(
      train_token_dataset,
      batch_size=dcfg.batch_size,
      sampler=train_sampler,
      num_workers=4,
      pin_memory=True
  )
  model=ELT_DiT(dcfg).to(device)
  model = DDP(model, device_ids=[LOCAL_RANK])
  
  optimizer = torch.optim.AdamW(
    list(model.parameters()),
    lr=dcfg.lr,
    weight_decay=dcfg.weight_decay,
    betas=(0.9, 0.96)
  )
  noise_scheduler=NoiseScheduler(dcfg)
  def warmup_fn(step):
      if step < dcfg.warmup_steps:
          return step / dcfg.warmup_steps
      return 1.0

  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
  global_step = 0
  total_steps=dcfg.total_epochs*len(train_loader)
  for epoch in range(dcfg.total_epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    for batch_latents, batch_labels in train_loader:
      batch_latents=batch_latents.to(device)
      #batch_labels=batch_labels.to(device)

      t=torch.randint(0, dcfg.T, (batch_latents.shape[0],)).to(device)
      epsilon=torch.randn_like(batch_latents)

      x_t=noise_scheduler.add_noise(batch_latents, t, epsilon)

      lambda_val=1.0-(global_step/total_steps)
      lambda_val=max(0.0, lambda_val)

      loss_total, L_int=diff_ILSD_train_step(model, x_t, t, epsilon, lambda_val, dcfg)
      optimizer.zero_grad()
      loss_total.backward()
      optimizer.step()
      scheduler.step()

      global_step+=1

      if RANK == 0 and global_step % 100==0:
        print(f"Epoch {epoch+1}/{dcfg.total_epochs} "
                    f"Step {global_step} "
                    f"Loss {loss_total.item():.4f} "
                    f"λ {lambda_val:.4f} "
                    f"L_int {L_int}")
    if RANK == 0 and epoch % 10 == 0:
          torch.save({
              'epoch': epoch,
              'model': model.module.state_dict(),
              'optimizer': optimizer.state_dict(),
              'step': global_step,
          }, f'checkpoint_epoch_diffusion{epoch}.pt')
  dist.destroy_process_group()
  
if __name__ == '__main__':
  main()
    
