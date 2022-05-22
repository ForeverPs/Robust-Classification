import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GradientConcealment(nn.Module):
    def __init__(self, w=1e20, epsilon=1e-8):
        super().__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, x):
        return torch.sin(x * self.w) * self.epsilon + x


class ResizedPaddingLayer(nn.Module):
    def __init__(self, l):
        super(ResizedPaddingLayer, self).__init__()
        self.l = np.random.choice(list(range(l, 224)), size=1)[0]
        pad1, pad3 = np.random.choice(list(range(224 - self.l)), size=2)
        pad2, pad4 = 224 - self.l - pad1, 224 - self.l - pad3
        self.padding = nn.ZeroPad2d(padding=(pad1, pad2, pad3, pad4))

    def forward(self, x):
        resize = F.interpolate(x, (self.l, self.l))
        resize = self.padding(resize)
        return resize