import torch

class Linear ( object ) :
    
    
    def __init__(self, nb_input, nb_output, activation):
        
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.activation = activation
    
        self.eta = 0.01
        self.params = torch.normal(0, 1, (self.nb_output, self.nb_input))
        self.b = torch.normal(0, 1, (1, self.nb_output))
        
        
    
    def forward ( self , * input_ ) :
        
        
        self.input = input_[0]
        self.s = torch.mm(self.params,self.input.t()).t() + self.b
        self.output = self.activation.forward(self.s)
        return self.output
    
    
    
    
    def backward ( self , * gradwrtoutput ) :
        
        gradwrtoutputTensor = gradwrtoutput[0].t()
        
        gradwrs = torch.mul(gradwrtoutputTensor, self.activation.backward(self.s))

        gradwrparams = torch.mm(gradwrs.t() , self.input)
        print(self.eta*gradwrparams)
        
        self.params -= self.eta*gradwrparams
        self.b -= self.eta * gradwrs
        
        gradwrtxl = torch.mm(self.params.t(), gradwrs.t())
        
        return gradwrtxl

    
    
    def param ( self ):
        return params, b
