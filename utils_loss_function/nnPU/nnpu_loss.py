import torch
import torch.nn.functional as F


def sigmoid_loss(out, y):
    loss = out.gather(1, (1 - y.unsqueeze(1)).long()).mean()
    return loss


class nnPULoss(torch.nn.Module):
    def __init__(self, class_prior):
        super().__init__()
        self.class_prior = class_prior

    def forward(self, logits, labels):
        soft_logits = F.softmax(logits, dim=-1)

        p_idx = torch.where(labels == 0)[0]
        u_idx = torch.where(labels == 1)[0]

        if len(p_idx) == 0:
            loss_pos = torch.tensor(0).cuda()
            loss_pos_neg = torch.tensor(0).cuda()
        else:
            loss_pos = sigmoid_loss(soft_logits[p_idx], labels[p_idx])
            loss_pos_neg = sigmoid_loss(soft_logits[p_idx], torch.ones(len(p_idx)).cuda())

        loss_unl = sigmoid_loss(soft_logits[u_idx], labels[u_idx])

        if torch.gt((loss_unl - self.class_prior * loss_pos_neg), 0):
            loss = self.class_prior * (loss_pos - loss_pos_neg) + loss_unl
        else:
            loss = self.class_prior * loss_pos_neg - loss_unl

        return loss


class uPULoss(torch.nn.Module):
    def __init__(self, class_prior):
        super().__init__()
        self.class_prior = class_prior

    def forward(self, logits, labels):
        soft_logits = F.softmax(logits, dim=-1)

        p_idx = torch.where(labels == 0)[0]
        u_idx = torch.where(labels == 1)[0]

        if len(p_idx) == 0:
            loss_pos = torch.tensor(0).cuda()
            loss_pos_neg = torch.tensor(0).cuda()
        else:
            loss_pos = sigmoid_loss(soft_logits[p_idx], labels[p_idx])
            loss_pos_neg = sigmoid_loss(soft_logits[p_idx], torch.ones(len(p_idx)).cuda())

        loss_unl = sigmoid_loss(soft_logits[u_idx], labels[u_idx])

        loss = self.class_prior * (loss_pos - loss_pos_neg) + loss_unl

        return loss


class ImbPULoss(torch.nn.Module):
    def __init__(self, class_prior):
        super().__init__()
        self.class_prior = class_prior

    def forward(self, logits, labels):
        soft_logits = F.softmax(logits, dim=-1)

        p_idx = torch.where(labels == 0)[0]
        u_idx = torch.where(labels == 1)[0]

        if len(p_idx) == 0:
            loss_pos = torch.tensor(0).cuda()
            loss_pos_neg = torch.tensor(0).cuda()
        else:
            loss_pos = sigmoid_loss(soft_logits[p_idx], labels[p_idx])
            loss_pos_neg = sigmoid_loss(soft_logits[p_idx], torch.ones(len(p_idx)).cuda())

        loss_unl = sigmoid_loss(soft_logits[u_idx], labels[u_idx])

        loss_neg = (loss_unl - self.class_prior * loss_pos_neg) / (1 - self.class_prior)

        if torch.gt((loss_unl - self.class_prior * loss_pos_neg), 0):
            loss = 0.5 * loss_pos + 0.5 * loss_neg
            loss = self.class_prior * (loss_pos - loss_pos_neg) + loss_unl
        else:
            loss = -1 * 0.5 * loss_neg

        return loss
