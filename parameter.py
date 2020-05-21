import torch

class Parameter(object):

    def __init__(self, size):
        self.value = torch.zeros(size)
        self.grad = torch.zeros(size)

    def set_value(self, value):
        self.value = value

    def zero_grad(self):
        self.grad.zero_()

    def add_grad(self, grad):
        self.grad += grad
