import torch
import torch.nn.functional as F
import torch.nn as nn
    
class EntropyLoss(nn.Module):
    def forward(self,outputs):
        pred_softmax = F.softmax(outputs, dim=1)
        ent_loss = -(pred_softmax * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        return ent_loss