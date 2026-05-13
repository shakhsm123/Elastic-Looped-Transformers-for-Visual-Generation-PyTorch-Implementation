import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizer import encode


def random_masking(tokens, cfg):
  mask_prob=random.uniform(0.5, 1.0)
  mask=torch.rand(tokens.shape).to(tokens.device) < mask_prob
  masked_tokens=torch.where(mask, cfg.mask_token_id, tokens)
  return masked_tokens, tokens,  mask

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
