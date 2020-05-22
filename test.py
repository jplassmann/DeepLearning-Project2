import torch
torch.set_grad_enabled(False)
torch.manual_seed(0)
import activation
import layer
import sequential
import loss
import math
import optimizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

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
    print("Please choose between")
    print("\t Sigmoïd Output Activation with Binary Cross Entropy Loss (0)")
    print("\t Tanh Output Activation with Mean Square Error Loss (1)")
    choice = int(input("[DEFAULT = 1] : ") or "1")
    print("")
    
    train_input = torch.rand((1000,2))
    train_target = torch.rand((1000,1))
    test_input = torch.rand((1000,2))
    test_target = torch.rand((1000,1))
    
    if choice == 0:
        epochs = int(input("Please input how many epochs you want to train over [DEFAULT = 6000] : ") or "6000")
        print("")
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
        ps = model.parameters()
        optim = optimizer.SGD(model.parameters(), lr=0.05, decay=500)
        #for plotting later
        levels = [0, 0.25, 0.5, 0.75, 1]
        #for accuracy computation
        sigmoid = True
     
    else:
        epochs = int(input("Please input how many epochs you want to train over [DEFAULT = 4000] : ") or "4000")
        print("")
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
        ps = model.parameters()
        optim = optimizer.SGD(model.parameters())
        #for plotting later
        levels = [-1, -0.5, 0, 0.5, 1]
        #for accuracy computation
        sigmoid = False
     
    #Normalization
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    for i in range(epochs):

        output = model(train_input)

        optim.zero_grad()
        gradwrrtxL = criterion.backward(output, train_target)

        model.backward(gradwrrtxL)

        optim.step()

        if i % 100 == 0:
            print(f"Epoch # {i} / Loss : {criterion.forward(output, train_target):.2f} / Train Accuracy [%]: {computation_accuracy(model, train_input, train_target, sigmoid)*100:.2f} / Test Accuracy [%]: {computation_accuracy(model, test_input, test_target, sigmoid)*100:.2f}")
    
    print("")
    choice_plot = int(input("Do you want to plot the results, YES = 1, NO = 0 [DEFAULT = 1] : ") or "1")
    
    if choice_plot == 1:
        print("")
        print("... plotting ...")
        print("")
        
        size = (1000, 1000)
        colorMap = torch.empty(size)

        for i in range(size[1]):
            input_ = torch.full((size[0], 2), i/1000)
            for j in range(size[0]):
                input_[j, 0] = j/1000
            input_.sub_(mu).div_(std)

            output = model.forward(input_).reshape((size[0]))
            colorMap[:, i] = output
            
        fig, ax = plt.subplots(facecolor='w')
        plt.xlabel('X Position', fontsize=20)
        plt.ylabel('Y Position', fontsize=20)
        #ax.legend(prop={'size': 20})
        fig.set_size_inches(10, 10)
        fig.set_dpi(100)
        plt.imshow(colorMap)
        plt.gca().invert_yaxis()
        y_vals = ax.get_yticks()
        ax.set_yticklabels(['{:.2f}'.format(y * 0.001) for y in y_vals])
        x_vals = ax.get_xticks()
        ax.set_xticklabels(['{:.2f}'.format(x * 0.001) for x in x_vals])
        plt.colorbar(ticks=levels)
        plt.savefig("plot.pdf",bbox_inches='tight')

        print(f"The plot was successfully saved in the same directory as 'test.py', it is called 'plot.pdf'.")
        print(f"The plot is a color map of the output of the model when passed coordinates in the first quadrant.")

    print("")
    print("*** End of file, thank you ! ***")