import torch

class Linear ( object ) :
    
    self.nb_input = 0
    self.nb_output = 0
    self.activation = None
    self.params = None
    self.b = None
    self.input = None
    self.s = None
    self.eta = 0.01
    
    def __init__(nb_input, nb_output, activation):
        
        self.nb_input = input_
        self.nb_output = output_
        self.activation = activation
    
        self.params = torch.normal(0, 1, (self.nb_ouput, self.nb_input))
        self.b = torch.normal(0, 1, (self.nb_output, 1))
    
    def forward ( self , * input_ ) :
        
        self.input = _input
        self.s = torch.mm(self.params,self.input)
        self.output = activation.forward(self.s + self.b)
        return self.output

    def backward ( self , * gradwrtoutput ) :
        
        
        gradwrs = torch.mm(gradwroutput,self.activation.bakward(self.s))
        
        gradwrparams = torch.mul(gradwrs , self.input)
        
        params -= eta*gradwrparams
        b -= eta * gradwrs
        return torch.mul(self.params, gradwrs)

    def param ( self ):
        return params, b
