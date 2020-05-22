import torch
import activation
import layer
import sequential
import loss
import math
import optimiser

def computation_accuracy(model, test_input, test_target, sigmoid=False):
    """ Computation of the accuracy between an input and a target set.
    
         Args:
            model:       PyTorch neural network model
            data_input:  Tensor of size N x D representing the input dataset
            data_target: Tensor of size N x 1 representing the target of the dataset
            sigmoid:     Boolean to indicate if we're using sigmoid activation (= True) or not (= False)

        Returns:
            Percentage of wrong classified sample.
    """

    output = model(test_input)
    
    if sigmoid==True:
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
    else:
        output[output >= 0] = 1
        output[output < 0] = -1
        
    goodValue = torch.full((len(output), 1), 0, dtype=torch.float32)
    goodValue[output == test_target] = 1
    return goodValue.sum()/len(goodValue)

if __name__=="__main__":
    
    print("")
    print("Mini Project 2 - Deep Learning (EE-559) - PILLONEL Ken (270852) - PLASSMANN Jeremy (273908)")
    print("")
    print("We recommend using default values by pressing the ENTER key.")
    print("")
    #We do not do sophisticated input validation because it is not the goal of this exercise
    epochs = int(input("Please input how many epochs you want to train over [DEFAULT = 4000] :") or "4000")
    print(("Please choose between") or "10000")
    print("\t Sigmoïd Output Activation with Binary Cross Entropy Loss (0)")
    print("\t Tanh Output Activation with Mean Square Error Loss (1)")
    choice = int(input("[DEFAULT = 1] :") or "1")
    print("")
    
    train_input = torch.rand((1000,2))
    train_target = torch.rand((1000,1))
    test_input = torch.rand((1000,2))
    test_target = torch.rand((1000,1))
    
    if choice == 0:
        criterion = loss.LossBCE()
        model = sequential.Sequential(
            layer.Linear(2, 25),
            activation.ReLU(),
            layer.Linear(25, 50),
            activation.ReLU(),
            layer.Linear(50, 50),
            activation.ReLU(),
            layer.Linear(50, 25),
            activation.ReLU(),
            layer.Linear(25, 1),
            activation.Sigmoid()
        )
        train_target[((train_input-0.5)**2).sum(1) < 1/(2*math.pi)] = 0
        train_target[((train_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1
        test_target[((test_input-0.5)**2).sum(1) < 1/(2*math.pi)] = 0
        test_target[((test_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1
        sigmoid = True
     
    else:
        criterion = loss.LossMSE()
        model = sequential.Sequential(
                    layer.Linear(2, 25),
                    activation.ReLU(),
                    layer.Linear(25, 50),
                    activation.ReLU(),
                    layer.Linear(50, 50),
                    activation.ReLU(),
                    layer.Linear(50, 25),
                    activation.ReLU(),
                    layer.Linear(25, 1),
                    activation.Tanh()
                )
        train_target[((train_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
        train_target[((train_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1
        test_target[((test_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
        test_target[((test_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1
        sigmoid = False
     
    #Normalization
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    ps = model.parameters()
    optim = optimiser.SGD(model.parameters())
    for i in range(epochs):

        output = model(train_input)

        optim.zero_grad()
        gradwrrtxL = criterion.backward(output, train_target)

        model.backward(gradwrrtxL)

        optim.step()


        if i % 100 == 0:
            print(f"Epoch # {i} / Loss : {criterion.forward(output, train_target):.2f} / Train Accuracy [%]: {computation_accuracy(model, train_input, train_target, sigmoid)*100:.2f} / Test Accuracy [%]: {computation_accuracy(model, test_input, test_target, sigmoid)*100:.2f}")


print("")
print("*** End of file, thank you ! ***")