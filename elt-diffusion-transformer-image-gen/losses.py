def diff_gt_loss(predicted_noise, actual_noise):
  loss=F.mse_loss(predicted_noise, actual_noise)
  return loss
def diff_distillation_loss(student_pred, teacher_pred):
  loss = F.mse_loss(student_pred, teacher_pred)
  return loss
def diff_ILSD_train_step(model, x_t, t, actual_noise, lambda_val, cfg):
  L_int=random.randint(cfg.L_min, cfg.L_max-1)
  teacher_pred=model(x_t, t, n_loops=cfg.L_max)
  student_pred=model(x_t, t, n_loops=L_int)
  
  loss_teacher=diff_gt_loss(teacher_pred, actual_noise)
  loss_student=diff_gt_loss(student_pred, actual_noise)
  loss_student_distillation=diff_distillation_loss(student_pred, teacher_pred.detach())
  loss_total=loss_teacher+lambda_val*loss_student+(1-lambda_val)*loss_student_distillation
  return loss_total, L_int
