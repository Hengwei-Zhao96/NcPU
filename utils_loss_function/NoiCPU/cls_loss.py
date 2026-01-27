import torch
import torch.nn.functional as F
import torch.nn as nn

class ClsLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.warm_up = args.warm_up
        self.loss_type = args.loss_type

    def forward(self, outputs, confidence, idxs, epoch):
        if self.loss_type == "ce":
            logsm_outputs = F.log_softmax(outputs, dim=1)
        elif self.loss_type == "tce":
            logsm_outputs = self.taylor_CE(outputs, 4)

        p_idx = idxs["p_idx"]
        u_idx = idxs["u_idx"]
        pse_n_idx = idxs["pse_n_idx"]
        
        if epoch >= self.warm_up:
            batchCof = self.confidence_update(confidence.clone(), pse_n_idx)
        else:
            batchCof = confidence.clone()

        log_final_outputs = logsm_outputs * batchCof
        log_sample_loss = -(log_final_outputs).sum(dim=1)

        p_loss = (log_sample_loss[p_idx]).sum() / (len(p_idx) + 1e-8)
        u_loss = (log_sample_loss[u_idx]).sum() / (len(u_idx) + 1e-8)

        average_loss = p_loss + u_loss

        return average_loss
    
    def taylor_CE(self, outputs, t):
        taylor_outputs = torch.zeros_like(outputs)
        for i in range(1, t + 1):
            taylor_outputs -= torch.pow(1 - torch.softmax(outputs, dim=1), i) / i # - is used to align the utlization of CE Loss
        return taylor_outputs
    
    def confidence_update(self, batchCof, pse_n_idx):

        batchCof[pse_n_idx, 0] = torch.zeros(len(pse_n_idx)).cuda()
        batchCof[pse_n_idx, 1] = torch.ones(len(pse_n_idx)).cuda()

        return batchCof