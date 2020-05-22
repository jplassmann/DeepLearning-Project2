""" The different kind of layers are defined here. Only the Linear layer is defined at the moment."""

import torch
import nnmodule
import parameter
import math

class Linear(nnmodule.NNModule):
    """ Class defining the linear layer"""

    def __init__(self, nb_input, nb_output):
        """ Initialize a linear layer 
        
            Args:
                nb_input:  length of the input
                nb_output: length of the output
        """
        
        self.nb_input = nb_input
        self.nb_output = nb_output

        self.eta = 0.01
        #self.params = torch.normal(0, 1, (self.nb_output, self.nb_input))
        self.params = parameter.Parameter((nb_output, nb_input))
        self.params.set_value(torch.empty(nb_output, nb_input).uniform_(
            -1/math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input)))
        # self.params = torch.empty(self.nb_output, self.nb_input).uniform_(
        #     -1/math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input))

        #self.b = torch.normal(0, 1, (1, self.nb_output))
        # self.b = torch.empty(1, self.nb_output).uniform_(
        #     -1 / math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input))
        self.b = parameter.Parameter((1, nb_output))
        self.b.set_value(torch.empty(1, nb_output).uniform_(
            -1/math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input)))

    def forward(self, *input_) :
        """ Compute the forward pass of the linear layer 
        
            Args: 
                input_: tuple with the input tensor in the first position
                
            Returns: output of the layer
        """
        
        self.input = input_[0]
        self.s = torch.mm(self.params.value, self.input.t()).t() + self.b.value
        return self.s

    def backward(self, *gradwrts) :
        """ Compute the backward pass of the linear layer. In this function,
            the gradient with respect to the weights are added to the Parameter.
        
            Args:
                gradwrts: tuple with the gradient with respect to s in first position
                
            Returns: gradient with respect to the output of the layer
        """
        
        gradwrtsTensor = gradwrts[0]

        gradwrparams = torch.mm(gradwrtsTensor.t() , self.input) / len(self.input)

        self.params.add_grad(gradwrparams)
        self.b.add_grad(gradwrtsTensor.sum(0) / len(self.input))

        gradwrtxl = torch.mm(gradwrtsTensor, self.params.value)

        return gradwrtxl

    def parameters(self):
        """ Returns: weights and biases of the layer."""
        
        return [self.params, self.b]
