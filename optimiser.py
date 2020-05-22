"""File containing the optimisers defines for the project"""


import parameter
import torch


class SGD(object):
    """Optimizer that uses stochastic gradient descent as update rule."""

    def __init__(self, parameters, lr=0.01):
        """Initializer.

        Args:
            parameters: List containing the parameters that should be updated
                by the optimizer
            lr: Learning rate of the gradient descent rule
        """
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Sets the gradients of all parameters to 0.

        Should be called before backpropagating the gradients.
        """
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        """Updates the parameters using a gradient descent rule.

        Should be called after backpropagating the gradients."""
        for param in self.parameters:
            param.value -= self.lr * param.grad
