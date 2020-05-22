import torch

class LossMSE(object):

    def forward(self, x, y):
        return ((x - y)**2).sum()/len(x)

    def backward(self, x, y):
        return 2 * (x - y)


class LossBCE(object):

    reg = 1e-15
    def forward(self, x, y):
        return torch.sum(-y * torch.log(x.clamp(min=LossBCE.reg)) - (1 - y) * torch.log((1 - x).clamp(min=LossBCE.reg)))/len(x)

    def backward(self, x, y):
        return - y / x.clamp(min=LossBCE.reg) + (1 - y) / (1 - x).clamp(min=LossBCE.reg)
