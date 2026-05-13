import torch
import torch.nn.functional as F
import math

def any_time_generate(model, head, cfg, n_loops, class_label, cfg_scale, device):
  model.eval()
  with torch.no_grad():
    x_0=torch.full((1, cfg.max_seq_len), cfg.mask_token_id, dtype=torch.long).to(device)
    K=cfg.sampling_steps
    seq_len=cfg.max_seq_len
    for k in range(K):

      mask_ratio=math.cos(math.pi/2 * k/K)
      num_to_unmask=int(seq_len*(1-mask_ratio))


      hidden_state=model(x_0, n_loops=n_loops)
      logits=head(hidden_state)
      softmax_logits=F.softmax(logits, dim=-1)
      softmax_logits=softmax_logits.squeeze(0)
      token_IDs=torch.multinomial(softmax_logits,num_samples=1)
      token_IDs=token_IDs.squeeze(-1)

      confidence=softmax_logits[torch.arange(seq_len), token_IDs]
      current_mask = (x_0.squeeze(0) == cfg.mask_token_id)
      confidence[~current_mask] = float('inf')

      if num_to_unmask > 0:
        _, top_indices = torch.topk(confidence, k=num_to_unmask, largest=True)
        x_0[0, top_indices] = token_IDs[top_indices]
      x_0 = x_0.clamp(0, cfg.vocab_size - 2)
  return x_0
