class LossMSE(object):


    def forward(self, x, y):
    
        return ((x - y)**2).sum()/len(x)
        
    

    def backward(self, x, y):
        
        return 2 * (x - y)