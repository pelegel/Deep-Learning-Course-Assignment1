# Assignment 1. MLPs and Backpropagation
#### BGU IEM Introduction to Deep Learning - 2023

The goal of the first sections in this assignment is to practice index notation as it is used in linear algebra. 
In the final section, vector calculus will be applied to an MLP in order to derive the equations of backpropagation for the basic modules in a vanilla neural network.

### Question 3 - NumPy implementation
In this assignment we will implement a multi-layer perceptron using purely NumPy routines. The network should consist of a series of linear layers with ReLU activation functions followed by a final linear layer and softmax activation. As a loss function, we will use the common cross-entropy loss for classification tasks. To optimize our network we will use the mini-batch stochastic gradient descent algorithm. The code is implemented in the files:
* train_mlp_numpy.py
* modules.py
* mlp_numpy.py

Part of the success of neural networks is the high efficiency on graphical processing units (GPUs) through matrix multiplications. Therefore, we will use matrix multiplications rather than iterating over samples in the batch or weight rows/columns. 
We will provide the achieved test accuracy and loss curve for the for the default values of parameters (one hidden layer, 128 hidden units, 10 epochs, learning rate 0.1).


### Question 4 - PyTorch MLP
We will implement the same MLP in pytorch by following the instructions inside the file:
* train_mlp_pytorch.py
* mlp_pytorch.py

The interface is similar to mlp_numpy.py.
Using the same parameters as in Question 3, we should get similar accuracy on the test set. We will provide the achieved test accuracy and loss curve for the default values of parameters (one layer, 128 hidden units, 10 epochs, no batch normalization, learning rate 0.1).

