import torch
import numpy as np
import activation
import layer
import sequential
import loss

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
    for p in model.parameters():
        print(p.size())
