import torch
import torch.nn as nn
import torch.nn.functional as F
class VectorQuantizer(nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding=nn.Embedding(1024, 256)
  def forward(self, z):
    B, _, _, _=z.shape
    z = z.permute(0, 2, 3, 1)
    z = z.reshape(-1, 256)


    distances = (
    z.pow(2).sum(dim=1, keepdim=True)
    + self.embedding.weight.pow(2).sum(dim=1)
    - 2 * z @ self.embedding.weight.t()
    )
    indices = distances.argmin(dim=1)
    indices = indices.reshape(B, 16, 16)
    quantized = self.embedding(indices)
    quantized = quantized.permute(0, 3, 1, 2)
    return indices, quantized

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels=in_channels
    self.out_channels=out_channels
    self.norm1=nn.GroupNorm(32, in_channels)
    #self.activation_1=F.silu() -> error
    self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.norm2=nn.GroupNorm(32, out_channels)
    self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
    self.nin_shortcut = None
    if in_channels != out_channels:
      self.nin_shortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    shortcut=x
    x=self.norm1(x)
    x = F.silu(x)
    x = self.conv1(x)
    #shortcut=x
    x=self.norm2(x)
    x=F.silu(x)
    x=self.conv2(x)

    if(self.in_channels!=self.out_channels):
      x = x + self.nin_shortcut(x)
    else:
      x=x+shortcut
    return x
class DownStage(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=True):
    super().__init__()
    self.block=nn.ModuleList([
        ResBlock(in_channels, out_channels),
        ResBlock(out_channels, out_channels)
    ])
    self.downsample=downsample

  def forward(self, x):
    for block in self.block:
      x=block(x)
    if self.downsample==True:
      x=F.avg_pool2d(x, kernel_size=2, stride=2)
    return x
class VQGANEncoder(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    self.in_channels=in_channels
    self.conv_in=nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
    self.down=nn.ModuleList([
        DownStage(128, 128, downsample=True),
        DownStage(128, 128, downsample=True),
        DownStage(128, 256, downsample=True),
        DownStage(256, 256, downsample=True),
        DownStage(256, 512, downsample=False),
    ])
    self.mid=nn.ModuleList([
        ResBlock(512,512),
        ResBlock(512,512),
    ])
    self.norm_out=nn.GroupNorm(32, 512)
    self.conv_out=nn.Conv2d(512,256, kernel_size=1)
  def forward(self, x):
    x=self.conv_in(x)
    for down in self.down:
      x=down(x)
    for res_block in self.mid:
      x=res_block(x)
    x=self.norm_out(x)
    x=F.silu(x)
    x=self.conv_out(x)
    return x

class UpStage(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=True):
    super().__init__()
    self.block=nn.ModuleList([
        ResBlock(in_channels, out_channels),
        ResBlock(out_channels, out_channels)
    ])
    self.upsample=upsample
    self.upsample_conv=None
    if upsample:
      self.upsample_conv=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
  def forward(self, x):
    for block in self.block:
      x=block(x)
    if self.upsample:
      x=F.interpolate(x, scale_factor=2, mode='nearest')
      x=self.upsample_conv(x)
    return x

class VQGANDecoder(nn.Module):
  def __init__(self, out_channels=3):
    super().__init__()
    self.conv_in=nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.mid=nn.ModuleList([
        ResBlock(512, 512),
        ResBlock(512, 512),
    ])
    self.up=nn.ModuleList([
        UpStage(128, 128, upsample=False),
        UpStage(256, 128, upsample=True),
        UpStage(256, 256, upsample=True),
        UpStage(512, 256, upsample=True),
        UpStage(512, 512, upsample=True),
    ])
    self.norm_out=nn.GroupNorm(32, 128)
    self.conv_out=nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
  def forward(self, x):
    x=self.conv_in(x)
    for res_block in self.mid:
      x=res_block(x)
    for counter in range(len(self.up)):
      x=self.up[len(self.up)-1-counter](x)
    x=self.norm_out(x)
    x=F.silu(x)
    x=self.conv_out(x)
    return x
class VQGAN(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder=VQGANEncoder()
    self.quantize=VectorQuantizer()
    self.decoder=VQGANDecoder()
  def encode(self, x):
    x=self.encoder(x)
    indices, quantized=self.quantize(x)
    return indices, quantized
  def decode(self, quantized):
    quantized=self.decoder(quantized)
    return quantized
  def forward(self, x):
    indices, quantized=self.encode(x)
    image_reconstructed=self.decode(quantized)
    return image_reconstructed, indices
def load_tokenizer(device):
  vqgan=VQGAN()
  state_dict=torch.load("ckpts/maskgit-vqgan-imagenet-f16-256.bin", map_location="cpu")
  vqgan.load_state_dict(state_dict, strict=False)
  vqgan.to(device)
  vqgan.eval()
  for param in vqgan.parameters():
      param.requires_grad = False
  print("tokenizer loaded")
  print(f"parameters: {sum(p.numel() for p in vqgan.parameters()):,}")
  return vqgan

def encode(vqgan, images):
    # images: (B, 3, 256, 256), values in [-1, 1]
    with torch.no_grad():
        indices, quantized = vqgan.encode(images)
        indices = indices.reshape(images.shape[0], -1)  # (B, 256)
    return indices

def decode(vqgan, indices):
    with torch.no_grad():
        batch = indices.shape[0]
        indices = indices.clamp(0, 1023)
        indices_2d = indices.reshape(batch, 16, 16)
        quantized = vqgan.quantize.embedding(indices_2d)
        quantized = quantized.permute(0, 3, 1, 2)
        images = vqgan.decode(quantized)
    return images
