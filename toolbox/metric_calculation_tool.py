import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def pu_metric(y_true, y_pred, y_score=None, pos_label=0):
    """
    inputs: torch.tensor
    outputs: {"key":torch.tensor}
    """
    metrics = dict()

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    oa = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label, zero_division=0)
    precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label, zero_division=0)

    metrics["OA"] = torch.tensor(oa) * 100
    metrics["F1"] = torch.tensor(f1) * 100
    metrics["Pre"] = torch.tensor(precision) * 100
    metrics["Rec"] = torch.tensor(recall) * 100

    if y_score is not None:
        y_score = y_score.cpu().numpy()
        if pos_label == 0:
            y_true = np.ones_like(y_true) - y_true
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        ap = average_precision_score(y_true=y_true, y_score=y_score)

        metrics["AUC"] = torch.tensor(auc) * 100
        metrics["AP"] = torch.tensor(ap) * 100

    return metrics


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k, while ignoring targets with label -1."""
    with torch.no_grad():
        valid_mask = target != -1
        if valid_mask.sum() == 0:
            return None
        
        output = output[valid_mask]
        target = target[valid_mask]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
