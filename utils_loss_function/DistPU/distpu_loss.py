from .distribution_loss import *
from .entropy_minimization import *


def DistPULoss(args):

    base_loss = LabelDistributionLoss(prior=args.class_prior)

    def loss_fn_entropy(outputs, labels):
        scores = torch.sigmoid(torch.clamp(outputs, min=-10, max=10))
        return base_loss(outputs, labels) + args.co_mu * loss_entropy(scores[labels != 1])

    if args.entropy == 1:
        return loss_fn_entropy

    return base_loss
