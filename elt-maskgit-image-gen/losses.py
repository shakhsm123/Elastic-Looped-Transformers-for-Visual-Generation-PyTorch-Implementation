import torch 
import torch.nn.functional as F
import random



def gt_loss(logits, targets, mask):
  logits=logits[mask]
  targets=targets[mask]
  loss=F.cross_entropy(logits, targets)
  return loss



def distillation_loss(student_logits, teacher_logits, mask):
  student_logits=student_logits[mask]
  teacher_logits=teacher_logits[mask]
  student_softmax=F.log_softmax(student_logits, dim=-1)
  teacher_softmax=F.softmax(teacher_logits, dim=-1)
  loss=F.kl_div(student_softmax, teacher_softmax, reduction='batchmean')
  return loss

def ILSD_train_step(model, head, tokens, targets, mask, lambda_val, cfg):
  L_int=random.randint(cfg.L_min, cfg.L_max-1)
  teacher_hidden_state=model(tokens, n_loops=cfg.L_max)
  teacher_logits=head(teacher_hidden_state)
  student_hidden_state=model(tokens, n_loops=L_int)
  student_logits=head(student_hidden_state)


  loss_teacher=gt_loss(teacher_logits, targets, mask)
  loss_student_gt=gt_loss(student_logits, targets, mask)
  loss_student_distillation=distillation_loss(student_logits, teacher_logits.detach(), mask)
  total=loss_teacher+lambda_val*loss_student_gt+(1-lambda_val)*loss_student_distillation

  return total, L_int
