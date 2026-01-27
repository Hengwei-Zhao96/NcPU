import torch.nn as nn
import torch

import numpy as np
import torch.nn.functional as F

class NoiseMatrixLayer(nn.Module):
    def __init__(self, num_classes, init=2):
        '''
        from https://arxiv.org/abs/2102.02400.
        '''
        super(NoiseMatrixLayer, self).__init__()
        self.num_classes = num_classes
        self.noise_layer = nn.Linear(self.num_classes, self.num_classes, bias=False).cuda()
        self.noise_layer.weight.data.copy_(-init * torch.ones(num_classes, num_classes))
        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.cuda()
        self.identity = torch.eye(num_classes).cuda()

    def forward(self):
        sig = torch.sigmoid(self.noise_layer(self.identity))
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)
        return T