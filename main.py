import torch
import numpy as np
import activation
import layer
import sequential
import loss
import math
import optimiser


def test_accuracy(model, test_input, test_target):

    output = model.forward(test_input)

    output[output >= 0] = 1
    output[output < 0] = -1
    goodValue = torch.full((len(output), 1), 0)
    goodValue[output == test_target] = 1
    return goodValue.sum()/len(goodValue)

if __name__=="__main__":

    loss = loss.LossMSE()
    model = sequential.Sequential(
                layer.Linear(2, 25),
                activation.ReLU(),
                layer.Linear(25, 25),
                activation.ReLU(),
                layer.Linear(25, 25),
                activation.ReLU(),
                layer.Linear(25, 1)
            )


    train_input = torch.rand((1000,2))
    train_target = torch.rand((1000,1))
    train_target[((train_input-0.5)**2).sum(1) < 1/(2*np.pi)] = -1
    train_target[((train_input-0.5)**2).sum(1) >= 1/(2*np.pi)] = 1

    test_input = torch.rand((1000,2))
    test_target = torch.rand((1000,1))
    test_target[((test_input-0.5)**2).sum(1) < 1/(2*np.pi)] = -1
    test_target[((test_input-0.5)**2).sum(1) >= 1/(2*np.pi)] = 1

    epochs = 4000
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
