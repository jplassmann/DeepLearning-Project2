import torch
import numpy as np
import activation
import layer
import sequential
import loss
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Optimiser works
=======
>>>>>>> b5a787b4618340bb03d5ef9efd504ff08192613f
import math
import optimiser


def test_accuracy(model, test_input, test_target):

    output = model.forward(test_input)

    output[output >= 0] = 1
    output[output < 0] = -1
    goodValue = torch.full((len(output), 1), 0)
    goodValue[output == test_target] = 1
    return goodValue.sum()/len(goodValue)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Changerd names and added optimiser and parameters
=======
>>>>>>> Optimiser works
=======
>>>>>>> b5a787b4618340bb03d5ef9efd504ff08192613f

if __name__=="__main__":

    loss = loss.LossMSE()
    model = sequential.Sequential(
                layer.Linear(2, 25),
                activation.ReLU(),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b5a787b4618340bb03d5ef9efd504ff08192613f
                layer.Linear(25, 50),
                activation.ReLU(),
                layer.Linear(50, 50),
                activation.ReLU(),
                layer.Linear(50, 25),
                activation.ReLU(),
                layer.Linear(25, 1),
                activation.Tanh()
            )


    train_input = torch.rand((1000,2))
    train_target = torch.rand((1000,1))
    train_target[((train_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
    train_target[((train_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1

    test_input = torch.rand((1000,2))
    test_target = torch.rand((1000,1))
    test_target[((test_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
    test_target[((test_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1

    #Normalization
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    epochs = 10000
    ps = model.parameters()
    optim = optimiser.SGD(model.parameters())
    for i in range(epochs):

        output = model.forward(train_input)

        optim.zero_grad()
        #print(ps[0].grad[0])
        gradwrrtxL = loss.backward(output, train_target)
        #print(ps[0].grad[0])
        model.backward(gradwrrtxL)
        #print(ps[0].grad[0])
        optim.step()
        #print(ps[0].grad[0])

        if i % 10 == 0:
            test_accuracyV = test_accuracy(model, test_input, test_target)

            print(loss.forward(output, train_target), test_accuracyV, test_accuracy(model, train_input, train_target))
<<<<<<< HEAD
=======
                layer.Linear(25, 25),
=======
                layer.Linear(25, 50),
>>>>>>> Optimisatino debug
                activation.ReLU(),
                layer.Linear(50, 50),
                activation.ReLU(),
                layer.Linear(50, 25),
                activation.ReLU(),
                layer.Linear(25, 1),
                activation.Tanh()
            )
<<<<<<< HEAD
    for p in model.parameters():
        print(p.size())
>>>>>>> Changerd names and added optimiser and parameters
=======


    train_input = torch.rand((1000,2))
    train_target = torch.rand((1000,1))
    train_target[((train_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
    train_target[((train_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1

    test_input = torch.rand((1000,2))
    test_target = torch.rand((1000,1))
    test_target[((test_input-0.5)**2).sum(1) < 1/(2*math.pi)] = -1
    test_target[((test_input-0.5)**2).sum(1) >= 1/(2*math.pi)] = 1

    #Normalization
    mu, std = train_input.mean(0), train_input.std(0)
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    epochs = 10000
    ps = model.parameters()
    optim = optimiser.SGD(model.parameters())
    for i in range(epochs):

        output = model.forward(train_input)

        optim.zero_grad()
        #print(ps[0].grad[0])
        gradwrrtxL = loss.backward(output, train_target)
        #print(ps[0].grad[0])
        model.backward(gradwrrtxL)
        #print(ps[0].grad[0])
        optim.step()
        #print(ps[0].grad[0])

        if i % 10 == 0:
            test_accuracyV = test_accuracy(model, test_input, test_target)

            print(loss.forward(output, train_target), test_accuracyV, test_accuracy(model, train_input, train_target))
>>>>>>> Optimiser works
=======
>>>>>>> b5a787b4618340bb03d5ef9efd504ff08192613f
