"""Class used to define parameters in a layer of the neural network."""


import torch


class Parameter(object):
    """Class used to define parameters used for the neural networks.

    This classes is used to store both the current value of the parameters as
    well as the accumulated gradient."""

    def __init__(self, size):
        """Initializer.

        Args:
            size: Size of the tensor that will hold the array
        """
        self.value = torch.zeros(size)
        self.grad = torch.zeros(size)

    def set_value(self, value):
        """Sets the value of the tensor, useful for initialzation.

        Args:
            value: Tensor to which the value of the parameter will be set."""
        self.value = value

    def zero_grad(self):
        """Sets the accumulated gradient to zero."""
        self.grad.zero_()

    def add_grad(self, grad):
        """Adds the provided tensor the the gradient held by the class.

        Args:
            grad: Tensor that is added to the cumulated gradient."""
        self.grad += grad
