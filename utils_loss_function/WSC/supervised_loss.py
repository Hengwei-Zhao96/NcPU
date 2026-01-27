import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedNoisyLoss(nn.Module):
    def __init__(self, balance):
        super(SupervisedNoisyLoss, self).__init__()
        self.num_classes = 2
        self.balance = balance
        
    def forward(self, noisy_probs, target):
        one_hot_target = F.one_hot(target, num_classes=self.num_classes)
        if self.balance:
            p_idx = torch.where(target == 0)[0]
            n_idx = torch.where(target == 1)[0]

            if len(p_idx) == 0:
                loss_pos = torch.tensor(0).cuda()
            else:
                loss_pos = torch.mean(-torch.sum(one_hot_target * torch.log(noisy_probs), dim=-1)[p_idx])

            loss_neg = torch.mean(-torch.sum(one_hot_target * torch.log(noisy_probs), dim=-1)[n_idx])

            noise_loss = (loss_pos + loss_neg) / 2.0
        else:
            noise_loss = torch.mean(
            -torch.sum(
                one_hot_target * torch.log(noisy_probs), dim=-1
            )
        )
        return noise_loss
    
def ce_loss(logits, targets, reduction='none'):
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)