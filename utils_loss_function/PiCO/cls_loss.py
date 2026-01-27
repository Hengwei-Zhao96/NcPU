import torch
import torch.nn.functional as F
import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self, confidence, positive_flag, unlabeled_flag, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence.cuda()
        self.positive_flag = positive_flag.cuda()
        self.unlabeled_flag = unlabeled_flag.cuda()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1.0 * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index, epoch):
        logsm_outputs = F.log_softmax(outputs, dim=1)

        p_flag = self.positive_flag[index]
        u_flag = self.unlabeled_flag[index]

        log_final_outputs = logsm_outputs * self.confidence[index, :]

        log_sample_loss = -(log_final_outputs).sum(dim=1)

        p_loss = (log_sample_loss * p_flag).sum() / (p_flag.sum() + 1e-8)
        n_loss = (log_sample_loss * u_flag).sum() / (u_flag.sum() + 1e-8)
        average_loss = p_loss + n_loss

        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
        return None
