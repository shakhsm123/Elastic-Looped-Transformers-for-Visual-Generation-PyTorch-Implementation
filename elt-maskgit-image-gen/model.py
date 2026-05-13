import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class ELTConfig:
# architeccture params
  d_model: int=512
  n_heads: int=8
  d_ff : int=2048
  dropout: float=0.1
  n_unique_layers: int=4
  max_seq_len: int=256
  vocab_size: int=1025
# loop params
  L_max: int=4
  L_min: int=1
# distillation params
  lambda_start: float=1.0
  lambda_end : float=0.0
# training params
  batch_size: int=64
  lr : float=1e-4
  weight_decay: float=4.5e-2
  warmup_steps: int=15000
  total_epochs: int=270
  label_drop_prob: float=0.1
# inference params
  sampling_steps: int=24
  cfg_scale: float=3.0
  @property
  def mask_token_id(self):
    return self.vocab_size - 1
cfg =cfg = ELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,   # 3 × 4 = 12 effective depth
    L_max=4,
    batch_size=128,
)


class LearnedPositionalEmbedding(nn.Module):
  def __init__(self, cfg: ELTConfig):
    super().__init__()
    self.pos_embed=nn.Embedding(cfg.max_seq_len, cfg.d_model)
  def forward(self, x):
    batch, seq_len=x.shape
    pos=torch.arange(seq_len, device=x.device)
    pos=self.pos_embed(pos)
    pos=pos.expand(batch, -1, -1)
    return pos



class TransformerBlock(nn.Module):
  def __init__(self, cfg: ELTConfig):
    super().__init__()
    self.layer_norm=nn.LayerNorm(cfg.d_model)
    self.attn=nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True, dropout=cfg.dropout)
    self.layer_norm2=nn.LayerNorm(cfg.d_model)
    self.ffn=nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_ff),
        nn.GELU(),
        nn.Linear(cfg.d_ff, cfg.d_model)
    )
    self.ffn_dropout=nn.Dropout(cfg.dropout)
  def forward(self, x, padding_mask=None):
    output=self.layer_norm(x)
    output, attention_weights=self.attn(output,output,output, key_padding_mask=padding_mask)
    output=self.ffn_dropout(output)
    x=x+output
    output=self.layer_norm2(x)
    output=self.ffn(output)
    output=self.ffn_dropout(output)
    x=x+output
    return x



class ELTModel(nn.Module):
  def __init__(self, cfg: ELTConfig):
    super().__init__()
    self.token_embedding=nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.pos_embedding=LearnedPositionalEmbedding(cfg)
    self.unique_layers=nn.ModuleList([TransformerBlock(cfg) for i in range(cfg.n_unique_layers)])
    self.layer_norm=nn.LayerNorm(cfg.d_model)
    self.dropout=nn.Dropout(cfg.dropout)
  def g_theta(self, x, padding_mask):
    for block in self.unique_layers:
      x=block(x, padding_mask)
    return x
  def forward(self,x, n_loops, padding_mask=None):
    tokens=x
    x=self.token_embedding(x)
    x=x+self.pos_embedding(tokens)
    x=self.dropout(x)
    for i in range(n_loops):
      x=self.g_theta(x, padding_mask)
    x=self.layer_norm(x)
    return x

class MLMHead(nn.Module):
  def __init__(self, cfg: ELTConfig):
    super().__init__()
    self.linear_projection=nn.Linear(cfg.d_model, cfg.vocab_size)
    self.layer_norm=nn.LayerNorm(cfg.d_model)
    self.activation=nn.GELU()
  def forward(self, x):
    x=self.layer_norm(x)
    x=self.activation(x)
    x=self.linear_projection(x)
    return x
