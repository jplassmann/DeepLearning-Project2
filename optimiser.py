import torch
import parameter

class SGD(object):

    def __init__(self, parameters, lr=0.005):
        self.parameters = parameters
        self.lr = lr


    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        for param in self.parameters:
            param.value -= self.lr * param.grad
