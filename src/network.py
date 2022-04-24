# % load network.py

"""
network.py

"""


import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(a, w)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {} : {}/{}'.format(j,
                      self.evaluate(test_data), n_test));
            else:
                print('Epoch {} complete'.format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x,y)``, and ``eta`` is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnw for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases,
                        nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all theactivations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose(())

        for l in range(2, self.num_layers):
            z=zs[-l]
            sp=sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

        def evaluate(self, test_data):
            """Return the number of test inputs for which the neural network outputs the correct result. Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
            """


        def cost_derivative(self, output_activations, y):
            """Return the vector of partial derivatives \partial C_x /
            \partial a for the output activations."""
            return (output_activation-y)

    # Miscellaneous functions
    def sigmoid(z):
        """The sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function"""
        return sigmoid(z)*(1-sigmoid(z))
