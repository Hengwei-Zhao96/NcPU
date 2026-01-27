import torch

def get_cont_mask(tau, logits, epoch, warm_up_epoch=1):
    
    soft_targets = torch.softmax(logits, dim=1)
    
    mask = torch.matmul(soft_targets, soft_targets.t()).clone().detach()
    if epoch < warm_up_epoch:
        mask = torch.zeros_like(mask)
    
    mask.fill_diagonal_(1)
    pos_mask = (mask >= tau).float()

    return pos_mask

def get_true_mask(true_label):
    pos_mask = torch.eq(true_label.view(-1,1), true_label.view(-1,1).T).float().cuda()
    return pos_mask

def get_weakly_supervised_mask(label):
    pos_mask = torch.eq(label.view(-1,1), label.view(-1,1).T).float().cuda()
    return pos_mask