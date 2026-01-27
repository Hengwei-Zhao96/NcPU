import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, balance=False):
        super(CELoss, self).__init__()
        self.balance = balance

    def forward(self, logits, labels):

        if self.balance:
            p_idx = torch.where(labels == 0)[0]
            n_idx = torch.where(labels == 1)[0]

            if len(p_idx) == 0:
                loss_pos = torch.tensor(0).cuda()
            else:
                loss_pos = nn.CrossEntropyLoss()(logits[p_idx], labels[p_idx])

            loss_neg = nn.CrossEntropyLoss()(logits[n_idx], labels[n_idx])

            loss = (loss_pos + loss_neg) / 2.0
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss
