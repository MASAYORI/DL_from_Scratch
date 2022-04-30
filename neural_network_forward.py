import numpy as np

from activation_function import *


def init_network():
    network = dict()
    network['W1'] = np.random.rand(2, 3)
    network['b1'] = np.random.rand(3)
    network['W2'] = np.random.rand(3, 2)
    network['b2'] = np.random.rand(2)
    network['W3'] = np.random.rand(2, 2)
    network['b3'] = np.random.rand(2)

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y


def main():
    network = init_network()
    x = np.random.randn(2)
    y = forward(network, x)
    print(y)


if __name__ == '__main__':
    main()
