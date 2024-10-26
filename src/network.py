"""
network.py
~~~~~~~~~~

Modified from original network.py.
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Improvements:
    - Full-matrix based mini-batch update
    - Improved weight initialisation
    - Cost function selection
    - Regularisation selection
    - monitor frequency

TODO:
    - Neuron options other than sigmoid
"""

#### Libraries
# Standard library
import sys
import random
import time
import json

# Third-party libraries
import numpy as np

from activation import *
from costfunc import *
from regularisation import *


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, regu=None):
        """Parameters:
        sizes: list of number of neurons in each layer, including input & output layers
        cost: cost function selection
        regu: regularisation selection (None means no regularisation)
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.regu = regu
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_freq=1,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if j % monitor_freq == 0 or j == epochs-1:
                if monitor_training_cost:
                    cost = self.total_cost(training_data, lmbda)
                    training_cost.append(cost)
                    print("Cost on training data: {}".format(cost))
                if monitor_training_accuracy:
                    accuracy = self.accuracy(training_data, convert=True)
                    training_accuracy.append(accuracy/n)
                    print("Accuracy on training data: {} / {}".format(
                        accuracy, n))
                if evaluation_data and monitor_evaluation_cost:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print("Cost on evaluation data: {}".format(cost))
                if evaluation_data and monitor_evaluation_accuracy:
                    accuracy = self.accuracy(evaluation_data)
                    evaluation_accuracy.append(accuracy/n_data)
                    print("Accuracy on evaluation data: {} / {}".format(
                        accuracy, n_data))
        return {'eval_cost': evaluation_cost, 'eval_acc': evaluation_accuracy,
                'train_cost': training_cost, 'train_acc': training_accuracy}

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        X = np.concat([dat[0] for dat in mini_batch], axis=1)
        Y = np.concat([dat[1] for dat in mini_batch], axis=1)
        nabla_b, nabla_w = self.backprop(X, Y)
        # update weights without regularisation
        if not self.regu:
            self.weights = [w-(eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                           for b, nb in zip(self.biases, nabla_b)]
            return
        # update weights with regularisation
        self.weights = [w-self.regu.update_weight(n, eta, lmbda, w) \
                        -(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, X, Y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        nbatch = X.shape[1]
        for b, w in zip(self.biases, self.weights):
            bmat = np.tile(b, reps=nbatch)
            z = np.dot(w, activation)+bmat
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass, start from output layer
        delta = self.cost.delta(zs[-1], activations[-1], Y)
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # find gradient till the 2nd layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)
        cost /= len(data)
        if self.regu:
            cost += sum(self.regu.get_cost(len(data), lmbda, w) for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__),
                "regularisation": str(self.regu.__name__) if self.regu else None,
                }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules["costfunc"], data["cost"])
    regu = getattr(sys.modules["regularisation"], data["regularisation"])
    net = Network(data["sizes"], cost=cost, regu=regu)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
