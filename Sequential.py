class Sequential ( object ) :
    
    def __init__(self, * layers):
        self.layers = layers
        
    
    def forward ( self , * input_ ) :
        x = input_[0]
        
        for l in self.layers:
            x = l.forward(x)
            
        self.output = x
        return self.output


    def backward ( self , * gradwrtoutput ) :
        
        gradwrtoutputTensor = gradwrtoutput[0]
        for l in self.layers[::-1]:
            
            gradwrtoutputTensor = l.backward(gradwrtoutputTensor)

    def param ( self ):
        return []
