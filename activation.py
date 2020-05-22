""" Activation functions are defined here """
import torch
import nnmodule

class ReLU(nnmodule.NNModule):
    """ ReLU activation function class """

    def forward(self, x):
        """ Forward pass of ReLU activation function 
        
            Args: 
                x: input tensor
        
            Returns: Tensor with the activation function applied to the input
        """
        
        self.s = x
        x[x < 0 ] = 0
        return x


    def backward(self, gradwrtoutput):
        """ Backward pass of ReLU activation function
        
            Args: 
                gradwrtoutput: gradient with respect to the output of the previous layer
                
            Returns: gradient with respect to self.s
        
        """
        self.s[self.s < 0] = 0
        self.s[self.s >= 0] = 1

        return torch.mul(gradwrtoutput,self.s)


class Tanh(nnmodule.NNModule):
    """ Tanh activation function class """

    def forward(self, x):
        """ Forward pass of Tanh activation function 
        
            Args: 
                x: input tensor
        
            Returns: Tensor with the activation function applied to the input
        """
        
        self.s = x
        return torch.tanh(self.s)


    def backward(self, gradwrtoutput):
        """ Backward pass of Tanh activation function
        
            Args: 
                gradwrtoutput: gradient with respect to the output of the previous layer
                
            Returns: gradient with respect to self.s
        
        """
        
        return torch.mul(gradwrtoutput, 1./(torch.cosh(self.s)**2))

class Sigmoid(nnmodule.NNModule):
    """ Sigmoid activation function class """

    def forward(self, x):
        """ Forward pass of Sigmoid activation function 
        
            Args: 
                x: input tensor
        
            Returns: Tensor with the activation function applied to the input
        """
        
        self.s = 1. / x.mul(-1).exp().add(1)
        return self.s


    def backward(self, gradwrtoutput):
        """ Backward pass of Sigmoid activation function
        
            Args: 
                gradwrtoutput: gradient with respect to the output of the previous layer
                
            Returns: gradient with respect to self.s
        
        """
        
        return torch.mul(gradwrtoutput, torch.mul(self.s,self.s.mul(-1).add(1)))
