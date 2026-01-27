class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', is_sum=False):
        self.name = name
        self.fmt = fmt
        self.is_sum = is_sum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if not self.is_sum:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({sum' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)