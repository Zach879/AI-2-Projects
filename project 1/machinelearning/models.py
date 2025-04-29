#Author: Zachary Reese
#File: models.py
#Date: 2/9/2025-2/22/2025
#Due-Date: 2/22/2025 @ 5am
#Course: CPSC548
#Professor: Dr. Schwesinger
#Assignment: Project 1
#Purpose: This file uses the nn.py defined classes and their helper methods to train a single layer (perceptron), two double layer (regression & digit classification), 
#           and a RNN (language classification) neural networks and tune their parameters accordingly.

import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w) #1 x dimensions

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        max_epochs = 10000
        for epoch in range(max_epochs):
            #iterate through the dataset one example at a time
            for x, y in dataset.iterate_once(1):
                #x = 1 x dimensions
                #y = 1 x 1
                true_label = y.data[0, 0]
                score = nn.as_scalar(self.run(x))
                if true_label * score < 1:  #hinge loss condition
                    update_direction = nn.Constant(true_label * x.data) #1 x dimensions
                    self.w.update(update_direction, 1.0)
            all_correct = True
            for x, y in dataset.iterate_once(1):
                #verify entire dataset correctly predicted with 100% accuracy.
                true_label = y.data[0, 0]
                if self.get_prediction(x) != true_label:
                    all_correct = False
                    break
            if all_correct:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.hidden_size = 50

        self.w1 = nn.Parameter(1, self.hidden_size) #1 x hidden_size
        self.b1 = nn.Parameter(1, self.hidden_size) #1 x hidden_size
        self.w2 = nn.Parameter(self.hidden_size, 1) #hidden_size x 1
        self.b2 = nn.Parameter(1, 1) #1 x 1

        self.learning_rate = 0.035

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #                        \/ x = batch_size x 1
        hidden_linear = nn.Linear(x, self.w1)  #batch_size x hidden_size
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)  #batch_size x hidden_size
        hidden = nn.ReLU(hidden_with_bias)  #batch_size x hidden_size
        out_linear = nn.Linear(hidden, self.w2)  #batch_size x 1
        output = nn.AddBias(out_linear, self.b2)  #batch_size x 1
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1)
        Returns: a loss node
        """
        pred = self.run(x) #batch_size x 1
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        threshold = 0.02  #stop training when average loss < threshold
        batch_size = 25   #batch size evenly divides the dataset of 200 examples
        max_epochs = 10000
        for epoch in range(max_epochs):
            losses = []  #accumulate loss values over an epoch
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                #x_batch = batch_size x 1
                #y_batch = batch_size x 1
                loss_node = self.get_loss(x_batch, y_batch)
                loss_val = nn.as_scalar(loss_node)
                losses.append(loss_val)
                gradients = nn.gradients(loss_node, [self.w1, self.b1, self.w2, self.b2])
                #Update parameters using gradient descent.
                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            avg_loss = sum(losses) / len(losses)
            #print("Epoch", epoch, "average loss:", avg_loss)
            if avg_loss < threshold:
                break
        #print("Final training loss:", avg_loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.
    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    """
    def __init__(self):
        self.hidden_size = 100

        self.W1 = nn.Parameter(784, self.hidden_size) #784 x hidden_size
        self.b1 = nn.Parameter(1, self.hidden_size) #1 x hidden_size
        self.W2 = nn.Parameter(self.hidden_size, 10) #hidden_size x 10
        self.b2 = nn.Parameter(1, 10) #1 x 10

        self.learning_rate = 0.05

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 784)
        Returns:
            A node with shape (batch_size x 10) containing predicted logits
        """
                               # \/ x = batch_size x 784
        hidden_linear = nn.Linear(x, self.W1)  #batch_size x hidden_size
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)  #batch_size x hidden_size
        hidden = nn.ReLU(hidden_with_bias)  #batch_size x hidden_size
        out_linear = nn.Linear(hidden, self.W2)  #batch_size x 10
        logits = nn.AddBias(out_linear, self.b2)  #batch_size x 10
        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10) (one-hot encoded)
        Returns: a loss node
        """
        logits = self.run(x)  #batch_size x 10
        return nn.SoftmaxLoss(logits, y) #Classify result into numeric loss value after converting logits to probabilities via softmax function.

    def train(self, dataset):
        batch_size = 25  #batch size evenly divides the 60,000 examples
        max_epochs = 10
        for epoch in range(max_epochs):
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                loss_node = self.get_loss(x_batch, y_batch)
                gradients = nn.gradients(loss_node, [self.W1, self.b1, self.W2, self.b2])
                #Update parameters using gradient descent.
                self.W1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.W2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            val_acc = dataset.get_validation_accuracy()
            #print("Epoch", epoch, "validation accuracy:", val_acc)
            if val_acc >= 0.97:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        self.hidden_size = 150

        self.W_input = nn.Parameter(self.num_chars, self.hidden_size) #47 x hidden_size; used in parameter sharing between feed-forward and RNN architectures.
        self.b_input = nn.Parameter(1, self.hidden_size) #1 x hidden_size

        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size) #hidden_size x hidden_size; recurrent weight step.

        self.W_out = nn.Parameter(self.hidden_size, len(self.languages)) #hidden_size x 5; hidden_size represents the number of words in batch.
        self.b_out = nn.Parameter(1, len(self.languages)) #1 x 5

        self.learning_rate = 0.1

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        #Process the first character using classic feed-forward network.
        #                 W_input = 47 x hidden_size
        #                 b_input = 1 x hidden_size
        #              \/ xs[0] = batch_size x 47
        z0 = nn.Linear(xs[0], self.W_input) #batch_size x hidden_size
        z0 = nn.AddBias(z0, self.b_input) #batch_size x hidden_size
        h = nn.ReLU(z0) #batch_size x hidden_size

        for x in xs[1:]: #Compute a projection of a middle char.
            #              W_input = 47 x hidden_size
            #              b_input = 1 x hidden_size
            #              x = batch_size x 47
            letter_proj = nn.Linear(x, self.W_input) #batch_size x hidden_size
            #              h = batch_size x hidden_size; gets recurrently updated.
            #              W_hidden = hidden_size x hidden_size
            hidden_proj = nn.Linear(h, self.W_hidden) #batch_size x hidden_size
            combined = nn.Add(letter_proj, hidden_proj) #batch_size x hidden_size; combine two projections.
            h = nn.ReLU(combined) #batch_size x hidden_size

        #Process the last char to the words' logits of batch_size length.
        #                  h = #batch_size x hidden_size
        #                  W_out = hidden_size x num_languages
        #                  b_out = 1 x num_languages
        logits_linear = nn.Linear(h, self.W_out) #batch_size x num_languages
        logits = nn.AddBias(logits_linear, self.b_out) #batch_size x num_languages
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        logits = self.run(xs)
        loss = nn.SoftmaxLoss(logits, y) #Classify result into numeric loss value after converting logits to probabilities via softmax function.
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 20  #Batch size must evenly divide the dataset of tens of thousands of examples.
        min_epochs = 3
        max_epochs = 20
        for epoch in range(max_epochs):
            #Iterate over the dataset one mini-batch at a time.
            for xs, y in dataset.iterate_once(batch_size):
                loss_node = self.get_loss(xs, y)
                loss_val = nn.as_scalar(loss_node)
                gradients = nn.gradients(loss_node, [self.W_input, self.b_input, self.W_hidden, self.W_out, self.b_out])
                #Update parameters using gradient descent.
                self.W_input.update(gradients[0], -self.learning_rate)
                self.b_input.update(gradients[1], -self.learning_rate)
                self.W_hidden.update(gradients[2], -self.learning_rate)
                self.W_out.update(gradients[3], -self.learning_rate)
                self.b_out.update(gradients[4], -self.learning_rate)
            
            val_acc = dataset.get_validation_accuracy()
            #print("Epoch", epoch, "average accuracy:", val_acc)
            if epoch >= min_epochs and val_acc >= 0.82: #goal is 81%, set to 82% to be safe.
                break

        #print("Final training accuracy:", val_acc)
