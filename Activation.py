import torch

    
class ReLU(object):

    def forward(self, x):
    
        self.s = x
        x[x < 0 ] = 0
        return x
    

    def backward(self, gradwrtoutput):
        
        self.s[self.s < 0] = 0
        self.s[self.s >= 0] = 1
        
        return torch.mul(gradwrtoutput,self.s) 