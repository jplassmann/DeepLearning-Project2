import torch

class Parameter(object):

    def __init__(self, size):
        self.parameter = torch.zeros(size)
        self.grad = torch.zeros(size)

    def zero_grad(self):
        self.grad.zero_()
