""" The different losses are defined here """

import torch

class LossMSE(object):
    """ Class defining the Mean Squared Error """
    
    def forward(self, x, y):
        """ Compute the MSE between the output of the model and the target
        
            Args: 
                x: output of the model
                y: target
            
            Returns: Mean Squared Error between the output of the model and the target
        """
        
        return ((x - y)**2).sum()/len(x)

    def backward(self, x, y):
        """ Compute the derivative of the loss with respect to the output
        
            Args: 
                x: output of the model
                y: target
                
            Returns: gradient with respect to the output
        """
        
        return 2 * (x - y)


class LossBCE(object):
    """" Class defining the Binary Cross Entropy loss """
    
    reg = 1e-15
    def forward(self, x, y):
        """ Compute the BCE between the output of the model and the target
        
            Args: 
                x: output of the model
                y: target
            
            Returns: BCE loss the output of the model and the target
        """
        
        return torch.sum(-y * torch.log(x.clamp(min=LossBCE.reg)) - (1 - y) * torch.log((1 - x).clamp(min=LossBCE.reg)))/len(x)

    def backward(self, x, y):
        """ Compute the derivative of the loss with respect to the output
        
            Args: 
                x: output of the model
                y: target
                
            Returns: gradient with respect to the output
        """
        
        return - y / x.clamp(min=LossBCE.reg) + (1 - y) / (1 - x).clamp(min=LossBCE.reg)
