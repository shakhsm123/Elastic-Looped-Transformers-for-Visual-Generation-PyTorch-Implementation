import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass(frozen=True)

class DiffusionELTConfig:
  #architecture hyperparams

  d_model: int=768
  n_heads: int=12
  d_ff: int=3072
  dropout: float = 0.1
  n_unique_layers: int=3
  L_max: int=4
  L_min: int=1

  #diffusion specific

  T:int=1000
  beta_start:float =0.0001
  beta_end:float=0.02

  #vae latent space specific

  latent_channels:int=4
  latent_size:int=32

  #training specific params
  batch_size: int=32
  lr:float=1e-4
  weight_decay:float = 4.5e-2
  warmup_steps:int=10000
  total_epochs:int=500

  #distillation
  lambda_start:float=1.0
  lambda_end:float=0.0

  #sampling
  sampling_steps:int=512

  @property
  def latent_seq_len(self):
    return self.latent_size * self.latent_size
  @property
  def patch_dim(self):
    return self.latent_channels


class LearnedPositionalEmbedding_Diffusion(nn.Module):
  def __init__(self, cfg: DiffusionELTConfig):
    super().__init__()
    self.pos_embed=nn.Embedding(cfg.latent_seq_len, cfg.d_model)
  def forward(self, x):
    batch, seq_len, _=x.shape
    pos=torch.arange(seq_len, device=x.device)
    pos=self.pos_embed(pos)
    pos=pos.expand(batch, -1, -1)
    return pos


class PatchEmbedding(nn.Module):
  def __init__(self,  cfg:DiffusionELTConfig):
    super().__init__()
    self.projection_layer=nn.Linear(cfg.patch_dim, cfg.d_model)
    self.pos_embedding=LearnedPositionalEmbedding_Diffusion(cfg)
    self.embedding_dropout=nn.Dropout(cfg.dropout)
  def forward(self, x):
    B=x.shape[0]
    x=x.reshape(B, -1, self.projection_layer.in_features)
    x=self.projection_layer(x)
    x=self.pos_embedding(x)+x
    x=self.embedding_dropout(x)
    return x


class TimeStepEmbedding(nn.Module):
  def __init__(self, cfg:DiffusionELTConfig):
    super().__init__()
    self.d_model=cfg.d_model
    self.linear_1=nn.Linear(self.d_model, self.d_model*4)
    self.activation=nn.SiLU()
    self.linear_2=nn.Linear(self.d_model*4, self.d_model)
  def forward(self, t):
    #sinusoidal encodings
    half=self.d_model//2
    freqs=torch.exp(-math.log(10000) * torch.arange(half)/half).to(t.device)
    args=t[:, None].float()*freqs[None]
    embedding=torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    embedding=self.linear_1(embedding)
    embedding=self.activation(embedding)
    embedding=self.linear_2(embedding)
    return embedding

class TransformerBlock(nn.Module):
  def __init__(self, cfg: DiffusionELTConfig):
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
    output, _=self.attn(output,output,output, key_padding_mask=padding_mask)
    output=self.ffn_dropout(output)
    x=x+output
    output=self.layer_norm2(x)
    output=self.ffn(output)
    output=self.ffn_dropout(output)
    x=x+output
    return x

class ELT_DiT(nn.Module):
  def __init__(self, cfg: DiffusionELTConfig):
    super().__init__()
    self.patch_embed=PatchEmbedding(cfg)
    self.time_embed=TimeStepEmbedding(cfg)
    self.looped_transformer=nn.ModuleList([TransformerBlock(cfg) for i in range(cfg.n_unique_layers)])
    self.norm=nn.LayerNorm(cfg.d_model)
    self.projection=nn.Linear(cfg.d_model, cfg.patch_dim)
  def g_theta(self, x, padding_mask=None):
    for unique_layer in self.looped_transformer:
      x=unique_layer(x, padding_mask)
    return x
  def forward(self, x, t, n_loops):
    patch_embed=self.patch_embed(x)
    t_embed=self.time_embed(t)
    t_embed=t_embed.unsqueeze(1)
    patch_embed+=t_embed
    x=patch_embed
    for i in range(n_loops):
      x=self.g_theta(x, padding_mask=None)
    x=self.norm(x)
    x=self.projection(x)
    B=x.shape[0]
    x=x.reshape(B, 4, 32, 32)
    return x
