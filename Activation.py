
class ReLU(object):

    def forward(self, x):
    
        x[x < 0 ] = 0
        return x
    

    def backward(self, x):
        
        
        x[x < 0] = 0
        x[x >= 0] = 1
        return x