import torch
import pandas as pd
import glob
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from vae import vae_encode


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
      
class LatentDataset(Dataset):
  def __init__(self, dataset, vae, device):
    temporary_dataloader=DataLoader(dataset, batch_size=32, shuffle=False)

    all_latents=[]
    all_labels=[]

    for image, labels in tqdm(temporary_dataloader):
      image=image.to(device)
      latent=vae_encode(vae, image)
      all_latents.append(latent.cpu())
      all_labels.append(labels.cpu())
    self.latents=torch.cat(all_latents, dim=0)
    self.labels=torch.cat(all_labels, dim=0)
  def __getitem__(self, idx):
    return self.latents[idx], self.labels[idx]
  def __len__(self):
    return len(self.latents)

class CachedLatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.latents)
