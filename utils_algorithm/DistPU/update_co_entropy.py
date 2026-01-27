import math


def update_co_entropy(args, epoch):
    co_entropy = (1 - math.cos((float(epoch) / args.pu_epochs) * (math.pi / 2))) * args.co_entropy
    return co_entropy
