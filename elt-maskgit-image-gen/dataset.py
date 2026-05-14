import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizer import encode

import pandas as pd
import glob
from PIL import Image
import io
def random_masking(tokens, cfg):
  mask_prob=random.uniform(0.5, 1.0)
  mask=torch.rand(tokens.shape).to(tokens.device) < mask_prob
  masked_tokens=torch.where(mask, cfg.mask_token_id, tokens)
  return masked_tokens, tokens,  mask


class ImageNet100Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # load all parquet files
        parquet_files = sorted(glob.glob(f"{data_dir}/train-*.parquet"))
        dfs = []
        for f in parquet_files:
            dfs.append(pd.read_parquet(f))
        self.df = pd.concat(dfs, ignore_index=True)
        self.transform = transform
        print(f"loaded {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row['label'])
        return image, label
      
class TokenDataset(Dataset):
  def __init__(self, dataset, tokenizer, device):
    temporary_dataloader=DataLoader(dataset, batch_size=32, shuffle=False)

    all_token_IDs=[]
    all_labels=[]

    for images, labels in tqdm(temporary_dataloader):
      images=images.to(device)
      indices=encode(tokenizer, images)
      all_token_IDs.append(indices.cpu())
      all_labels.append(labels.cpu())
    self.tokens = torch.cat(all_token_IDs, dim=0)
    self.labels=torch.cat(all_labels, dim=0)
  def __getitem__(self, idx):
    return self.tokens[idx], self.labels[idx]
  def __len__(self):
    return len(self.tokens)
    
class CachedTokenDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.tokens)
