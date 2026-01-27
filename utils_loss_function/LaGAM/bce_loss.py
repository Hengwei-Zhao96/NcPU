import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, label, weight=None):
        preds = torch.sigmoid(preds)
        logits_ = torch.cat([1.0 - preds, preds], dim=1)
        logits_ = torch.clamp(logits_, 1e-4, 1.0 - 1e-4)

        loss_entries = (-label * logits_.log()).sum(dim=0)
        label_num_reverse = 1.0 / label.sum(dim=0)
        loss = (loss_entries * label_num_reverse).sum()
        return loss
