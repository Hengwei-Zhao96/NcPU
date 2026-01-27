import torch.nn.functional as F


def mixup_bce(scores, targets_a, targets_b, lam):
    mixup_loss_a = F.binary_cross_entropy(scores, targets_a)
    mixup_loss_b = F.binary_cross_entropy(scores, targets_b)

    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b
    return mixup_loss
