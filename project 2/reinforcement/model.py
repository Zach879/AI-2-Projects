#Author: Zachary Reese
#File: model.py
#Date: 3/1/25-3/14/25
#Due-Date: 3/14/2025 @ 5am
#Course: CPSC548
#Professor: Dr. Schwesinger
#Assignment: Project 2

import nn
import numpy as np

class DeepQNetwork(object):
    def __init__(self, state_dim, action_dim):
        self.hidden_size = 150

        #Initialize parameters using the nn.Parameter class
        self.W1 = nn.Parameter(state_dim, self.hidden_size)    #state_dim x hidden_size
        self.b1 = nn.Parameter(1, self.hidden_size)            #1 x hidden_size
        self.W2 = nn.Parameter(self.hidden_size, action_dim)   #hidden_size x action_dim
        self.b2 = nn.Parameter(1, action_dim)                  #1 x action_dim

        self.parameters = [self.W1, self.b1, self.W2, self.b2]

        self.learning_rate = 0.4        #Learning rate for gradient updates.
        self.numTrainingGames = 5000     #Number of training games (> 1000).
        self.batch_size = 50

    def run(self, x):
        """
        Forward pass through the network.
        Input:
          x: a node with shape (batch_size x state_dim)
        Output:
          A node with shape (batch_size x action_dim) representing Q-values.
        """
        hidden_linear = nn.Linear(x, self.W1)                  # (batch_size x hidden_size)
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)  # (batch_size x hidden_size)
        hidden_activation = nn.ReLU(hidden_with_bias)          # (batch_size x hidden_size)
        q_pred = nn.Linear(hidden_activation, self.W2)         # (batch_size x action_dim)
        output = nn.AddBias(q_pred, self.b2)                   # (batch_size x action_dim)
        return output

    def get_loss(self, x, Q_target):
        """
        Computes the square loss between predicted Q-values and Q_target.
        Inputs:
          x: a node with shape (batch_size x state_dim)
          Q_target: a node with shape (batch_size x action_dim)
        Output:
          A loss node.
        """
        pred = self.run(x)  #(batch_size x action_dim)
        return nn.SquareLoss(pred, Q_target)

    def gradient_update(self, x, Q_target):
        """
        Performs a single gradient update using the gradients computed on the loss.
        Inputs:
          x: a node with shape (batch_size x state_dim)
          Q_target: a node with shape (batch_size x action_dim)
        Output:
          None
        """
        loss_node = self.get_loss(x, Q_target)
        gradients = nn.gradients(loss_node, [self.W1, self.b1, self.W2, self.b2])

        #Update the parameters using gradient descent.
        for param, grad in zip(self.parameters, gradients):
            param.update(grad, -self.learning_rate)
