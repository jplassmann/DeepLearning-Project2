class LossMSE(object):


    def forward(self, x, y):
    
        return ((x - y)**2).sum()/len(x)
        
    

    def backward(self, x, y):
        
        return 2 * (x - y)

class LossBCE(object):


    def forward(self, x, y):
        
        return - y * x.log() - ( 1 - y ) * torch.log(1-x)
        
    

    def backward(self, x, y):
        
        return - y / x + ( 1 - y ) / (1 - x )

