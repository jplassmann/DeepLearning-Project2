"""Sequential class used to build neural network models."""


class Sequential (object) :

    def __init__(self, *layers):
        """Initializer.

        Args:
            layers: Sequence of layers that are in the desired network."""
        self.layers = layers

    def __call__(self, input):
        """Overrides the call method to call the forward pass of the network.

        Args:
            input: Tensor of N * D containing the N samples.

        Returns:
            The output of the model.
        """
        return self.forward(input)

    def forward(self , input):
        """Performs the forward pass of the network.

        Args:
            input: Tensor of N * D containing the N samples. D is the number of
                features per sample

        Returns:
            A N * C tensor, the output of the model for the provided samples. C
            is the numbers of elements outputed by the neural network.
        """
        x = input
        for l in self.layers:
            x = l.forward(x)

        self.output = x
        return self.output

    def backward(self, gradwrtoutput) :
        """Backpropagates the gradient of the loss throughout the network.

        Args:
            gradwrtoutput: Tensor holding the gradient of the loss with respect
                to the output of the network.
            """
        for l in self.layers[::-1]:
            gradwrtoutput = l.backward(gradwrtoutput)

    def parameters(self):
        """Returns a list containing all the parameter tensors of the model."""
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
