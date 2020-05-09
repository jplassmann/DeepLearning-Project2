
class ReLU(object):

    def forward(x):
    
        x[x < 0 ] = 0
        return x
    

    def backward(x):
        
        
        x[x < 0] = 0
        x[x >= 0] = 1
        return x