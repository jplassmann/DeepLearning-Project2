
class LossMSE(object):


    def forward(x, y):
    
        return sum((x - y)**2)/len(x)
        
    

    def backward(x):
        
        
        