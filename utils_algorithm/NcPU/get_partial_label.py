import torch

def get_partial_label(one_hot_label):
    partial_label = torch.ones((len(one_hot_label),2))
    p_idxs = torch.where(one_hot_label == 0)[0]
    partial_label[p_idxs, 1] = 0
    return partial_label