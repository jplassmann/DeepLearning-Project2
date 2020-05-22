"""File containing the optimisers defines for the project"""

import parameter
import torch


class SGD(object):
    """Optimizer that uses stochastic gradient descent as update rule."""

    def __init__(self, parameters, lr=0.015, decay=1000):
        """Initializer.

        Args:
            parameters: List containing the parameters that should be updated
                        by the optimizer
            lr:         Learning rate of the gradient descent rule
            decay:      Decay rate of the learning rate
        """    
        self.parameters = parameters
        self.lr = lr
        self.decay = decay
        self.step_cnt = 0

    def zero_grad(self):
        """Sets the gradients of all parameters to 0.

        Should be called before backpropagating the gradients.
        """
        for param in self.parameters:
            param.zero_grad()
     
    def step(self):
        """Updates the parameters using a gradient descent rule.

        Should be called after backpropagating the gradients. The learning rate
        decreases each time step is called."""
        eta = self.lr/(1+self.step_cnt/self.decay)
        if eta < 0.001:
            eta = 0.001
        for param in self.parameters:
            param.value -= eta * param.grad
        self.step_cnt += 1
