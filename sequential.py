class Sequential (object) :

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, input_):
        return self.forward(input_)

    def forward(self , *input_) :
        x = input_[0]

        for l in self.layers:
            x = l.forward(x)

        self.output = x
        return self.output


    def backward(self, *gradwrtoutput) :

        gradwrtoutputTensor = gradwrtoutput[0]
        for l in self.layers[::-1]:

            gradwrtoutputTensor = l.backward(gradwrtoutputTensor)

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
