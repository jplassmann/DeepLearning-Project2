"""Superclass of different modules for the neural networks."""


class NNModule(object):
    """Holds the properties common to all types of modules.

    This class is used as a superclass for all the different modules that
    compose the neural network such as activation function and the neuron
    layers."""

    def parameters(self):
        """Returns an array of parameters that is empty by default.
        """
        return []
