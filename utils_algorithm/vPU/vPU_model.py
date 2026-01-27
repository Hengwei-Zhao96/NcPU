import torch.nn as nn


class vPU(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder
        self.LogSoftMax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h = self.base_encoder(x)
        return self.LogSoftMax(h)
