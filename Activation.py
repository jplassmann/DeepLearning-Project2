import torch
import nnmodule

class ReLU(nnmodule.NNModule):

    def forward(self, x):

        self.s = x
        x[x < 0 ] = 0
        return x


    def backward(self, gradwrtoutput):

        self.s[self.s < 0] = 0
        self.s[self.s >= 0] = 1

        return torch.mul(gradwrtoutput,self.s)

class Tanh(object):

    def forward(self, x):

        self.s = x
        return torch.tanh(self.s)


    def backward(self, gradwrtoutput):

        return torch.mul(gradwrtoutput, 1./(torch.cosh(self.s)**2))
