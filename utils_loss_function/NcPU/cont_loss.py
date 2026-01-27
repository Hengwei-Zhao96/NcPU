import torch
import torch.nn as nn

class NoiSNCL(nn.Module):
    def __init__(self):
        super(NoiSNCL, self).__init__()

    def forward(self, output_online, output_target, mask):

        batch_size = output_online.size(0)

        output_online = torch.clamp(output_online, 1e-4, 1.0 - 1e-4)
        output_target = torch.clamp(output_target, 1e-4, 1.0 - 1e-4)
        cos_sim = torch.matmul(output_online, output_target.t()) # B x B
        # <online, target> + sqrt(1-<online, target>)
        cos_sim = (1-1*cos_sim) * torch.zeros(batch_size, batch_size).fill_diagonal_(1).cuda() + ((1 - cos_sim).sqrt()) * torch.ones(batch_size, batch_size).fill_diagonal_(0).cuda()
        cont_logits = 2 * cos_sim
        
        loss_cont = ((cont_logits * mask).sum(dim=1)/mask.sum(1)).mean(0)

        return loss_cont