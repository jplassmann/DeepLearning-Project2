# DeepLearning - Project #2 - Mini Deep Learning Library

## Libraries used: 

<ul>
<li>Pytorch</li>
<li>Matplotlib</li>
</ul>

## File Description

<ul>
<li>activation.py: contains classes of activation functions implemented</li>
<li>layer.py: contains classes defining the layers</li>
<li>loss.py: contains classes defining the losses</li>
<li>nnmodule.py: parent class</li>
<li>optimizer.py: contains classes defining optimizers</li>
<li>parameter.py: contains the class Parameter</li>
<li>sequential.py: contain the class Sequential</li>
<li>test.py: contains an example of the test of this library</li>
</ul>

## Functionalities

### Layer

Only the linear fully connected layer has been implemented.
The linear layer can be created in the following way: Linear(nb_input, nb_output).

### Activation

ReLU, TanH and Sigmoid have been implemented.
The activation function can be created in the following way: ReLU(), Tanh(), Sigmoid().

### Losses 
<ul>
<li>Mean Squared Error: MSELoss()</li>
<li>Binary Cross Entropy: BCELoss()</li>
</ul>

Loss classes have two functions: 
<ul>
<li>forward(output, target) which computes the loss of the model.</li>
<li>backward(output, target) which computes the gradient of the loss with respect to the output of the model.</li>
</ul>

### Models

To create a model Sequential can be used in the following way: Sequential(layer1, activation1, ..., layern, activationn).
Three functions are available from that class: 
<ul>
<li>forward(input) which computes the output of the model given the input.</li>
<li>backward(gradwrtoutput) which triggers the backpropagation algorithm through all the layers of the model.</li>
</ul>

### Optimizers

Only the SGD optimizer has been implemented.
It can be created in the following way: SGD(model.parameters()).
<ul>
<li>zero_grad() set to zero the gradient of the parameters of the models</li>
<li>step() function updates the paramaters according to the gradient holded in the Parameter object.</li>
</ul>

## Execution

To run the program how it is intended you need to execute test.py.
You will first be asked to choose between two models. Just make your choice by inputing a 0 or 1 and pressing ENTER. You can also just input the ENTER key every time you're asked a choice to choose the default value. You will be then be asked how many epochs you want to train over. In the end, it will also ask you if you want to plot the results.