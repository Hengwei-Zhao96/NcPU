import numpy as np
import torch
import torch.nn.functional as F


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def three_sigma(x):

    idx = np.where(x < 0.2 / 9)
    return x[idx]


def loss_ft(args, logits1_u, logits1_u_s, targets_u, targets_p, epoch):
    label_u = F.one_hot(targets_u, 2).float().cuda()
    label_p = F.one_hot(targets_p, 2).float().cuda()
    lamda = (epoch / args.ft_epochs) ** 0.8
    label = lamda * label_p + (1 - lamda) * label_u
    loss = F.cross_entropy(logits1_u, label, reduction="mean")

    pseudo_label = torch.softmax(logits1_u.detach(), dim=-1)
    max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(0.9).float()
    loss2 = (F.cross_entropy(logits1_u_s, pseudo_targets_u, reduction="none") * mask).mean()
    return loss + loss2
