import torch
import nnmodule

class Linear(nnmodule.NNModule):


    def __init__(self, nb_input, nb_output):

        self.nb_input = nb_input
        self.nb_output = nb_output

        self.eta = 0.01
        #self.params = torch.normal(0, 1, (self.nb_output, self.nb_input))
        self.params = torch.empty(self.nb_output, self.nb_input).uniform_(-1/math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input))

        #self.b = torch.normal(0, 1, (1, self.nb_output))
        self.b = torch.empty(1, self.nb_output).uniform_(-1/math.sqrt(self.nb_input), 1/math.sqrt(self.nb_input))


    def forward ( self , * input_ ) :


        self.input = input_[0]
        self.s = torch.mm(self.params,self.input.t()).t() + self.b
        return self.s




    def backward ( self , * gradwrts ) :

        gradwrtsTensor = gradwrts[0]

        gradwrparams = torch.mm(gradwrtsTensor.t() , self.input)/len(self.input)

        self.params -= self.eta*gradwrparams
        self.b -= self.eta * gradwrtsTensor.sum(0)/len(self.input)

        gradwrtxl = torch.mm(gradwrtsTensor, self.params)

        return gradwrtxl



    def parameters(self):
        return [self.params, self.b]
