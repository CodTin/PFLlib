import torch.nn as nn


class BaseHeadSplit(nn.Module):
    def __init__(self, base: nn.Module, head: nn.Module):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out
