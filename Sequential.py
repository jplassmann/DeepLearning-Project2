class Sequential ( object ) :
    
    self.layers = []
    self.loss = None
    
    def __init__(loss, *layers):
        self.loss = loss
        self.layers = layers
        
    
    def forward ( self , * input_ ) :
        x = input_
        for l in layers:
            x = l.forward(x)
            
        return x


    def backward ( self , * gradwrtoutput ) :
        
        gradwrouput = loss.backward()
        
        for l in layers[:-1:-1]:
            
            gradwrouput = l.backward(gradwroutput)

    def param ( self ):
        return []
