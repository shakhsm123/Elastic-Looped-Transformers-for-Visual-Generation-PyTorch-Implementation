import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from model import ELTConfig, ELTModel, MLMHead
from tokenizer import load_tokenizer, decode
from inference import any_time_generate

device = torch.device('cuda:0')

cfg = ELTConfig(
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_unique_layers=3,
    L_max=4,
    batch_size=32,
)

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_epoch_10.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')
print(f"loaded epoch {checkpoint['epoch']} step {checkpoint['step']}")

model = ELTModel(cfg).to(device)
head = MLMHead(cfg).to(device)
model.load_state_dict(checkpoint['model'])
head.load_state_dict(checkpoint['head'])
model.eval()
head.eval()

vqgan = load_tokenizer(device)

loop_counts = [1, 2, 3, 4]
generated_images = []

for n_loops in loop_counts:
    token_ids = any_time_generate(
        model, head, cfg, n_loops=n_loops,
        class_label=None, cfg_scale=1.0, device=device
    )
    image = decode(vqgan, token_ids)
    image = image.squeeze(0).permute(1, 2, 0)
    image = (image + 1) / 2
    image = image.cpu().detach().numpy()
    image = np.clip(image, 0, 1)
    generated_images.append(image)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Any-Time Inference', fontsize=14)
for ax, img, n_loops in zip(axes, generated_images, loop_counts):
    ax.imshow(img)
    ax.set_title(f'L={n_loops}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('generated.png', dpi=150, bbox_inches='tight')
print("saved to generated.png")
