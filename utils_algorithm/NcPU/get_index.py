import torch

def get_pun_index(input, labels, threshold, softmax):

    idxs = {}

    if softmax:
        cls_pro = torch.softmax(input, dim=1)
    else:
        cls_pro = input.clone().detach()

    idxs["p_idx"] = torch.where(labels == 0)[0]
    idxs["u_idx"] = torch.where(labels == 1)[0]

    idxs["pse_n_idx"] = torch.where((labels == 1) & (cls_pro[:, 1] >= threshold[1]))[0]
    idxs["pse_p_idx"] = torch.where((labels == 1) & (cls_pro[:, 0] >= threshold[0]))[0]

    return idxs