import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

from model import ELTConfig, ELTModel, MLMHead
from tokenizer import load_tokenizer
from dataset import TokenDataset, CachedTokenDataset, random_masking, ImageNet100Dataset
from losses import ILSD_train_step

def main():
  LOCAL_RANK=int(os.environ.get("LOCAL_RANK"))
  RANK=int(os.environ.get("RANK"))
  WORLD_SIZE=int(os.environ.get("WORLD_SIZE"))

  dist.init_process_group(backend='nccl')

  torch.cuda.set_device(LOCAL_RANK)
  device=torch.device(f'cuda:{LOCAL_RANK}')

  cfg=ELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,
    L_max=4,
    batch_size=32,
  )
    
  token_cache = "/media/gpuhead-1/gpu-head-1-2nd/imagenet100_tokens_train.pt"


  if RANK == 0:
      if not os.path.exists(token_cache):
          print("rank 0: encoding dataset...")
          vqgan = load_tokenizer(device)
        
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
          token_dataset = TokenDataset(train_dataset, vqgan, device)
          torch.save({
              'tokens': token_dataset.tokens,
              'labels': token_dataset.labels
          }, token_cache)
          print("rank 0: cache saved")

  dist.barrier()

  cache = torch.load(token_cache, map_location='cpu')
  train_token_dataset = CachedTokenDataset(cache['tokens'], cache['labels'])

  
  train_sampler = DistributedSampler(
      train_token_dataset,
      num_replicas=WORLD_SIZE,
      rank=RANK,
      shuffle=True
  )
  train_loader = DataLoader(
      train_token_dataset,
      batch_size=cfg.batch_size,
      sampler=train_sampler,
      num_workers=4,
      pin_memory=True
  )
  model = ELTModel(cfg).to(device)
  head = MLMHead(cfg).to(device)
  model = DDP(model, device_ids=[LOCAL_RANK])
  head = DDP(head, device_ids=[LOCAL_RANK])

  optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(head.parameters()),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.96)
  )
  def warmup_fn(step):
      if step < cfg.warmup_steps:
          return step / cfg.warmup_steps
      return 1.0
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
  global_step = 0
  total_steps = cfg.total_epochs * len(train_loader)

  for epoch in range(cfg.total_epochs):
      train_sampler.set_epoch(epoch)
      model.train()
      head.train()
    
      for batch_tokens, batch_labels in train_loader:
          batch_tokens = batch_tokens.to(device)
        
          lambda_val = max(0.0, 1.0 - global_step / total_steps)
          masked_tokens, tokens, mask = random_masking(batch_tokens, cfg)
        
          loss, L_int = ILSD_train_step(
              model, head, masked_tokens,
              batch_tokens, mask, lambda_val, cfg
          )
        
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          scheduler.step()
          global_step += 1
        
          if RANK == 0 and global_step % 100 == 0:
              print(f"Epoch {epoch+1}/{cfg.total_epochs} "
                    f"Step {global_step} "
                    f"Loss {loss.item():.4f} "
                    f"λ {lambda_val:.4f} "
                    f"L_int {L_int}")
    
      if RANK == 0 and epoch % 10 == 0:
          torch.save({
              'epoch': epoch,
              'model': model.module.state_dict(),
              'head': head.module.state_dict(),
              'optimizer': optimizer.state_dict(),
              'step': global_step,
          }, f'checkpoint_epoch_{epoch}.pt')

  dist.destroy_process_group()

if __name__ == '__main__':
  main()
