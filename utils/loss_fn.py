import torch
import torch.nn.functional as F


def cross_entropy_loss_fn(logits, targets, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(targets * log_probs, dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def focal_loss_fn(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = cross_entropy_loss_fn(logits, targets, reduction=None)
    p_t = torch.exp(-ce_loss)
    return (alpha * (1 - p_t) ** gamma * ce_loss).mean()


def l1_loss_fn(outputs, targets, is_allzero=None):
    return F.l1_loss(outputs, targets, reduction="mean")


def l2_loss_fn(outputs, targets, is_allzero=None):
    return F.mse_loss(outputs, targets, reduction="mean")


def cos_loss_fn(outputs, targets, is_allzero=None):
    batch_size = outputs.size(0)
    loss = 1 - F.cosine_similarity(outputs.view(batch_size, -1), targets.view(batch_size, -1), dim=-1)
    if is_allzero is None:
        return loss.mean()

    valid = ~is_allzero.to(outputs.device)
    if not torch.any(valid):
        return outputs.new_tensor(0.0)
    return loss[valid].mean()
