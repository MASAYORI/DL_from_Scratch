import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle

from activation_function import *


def get_data():
    mnist_data = mnist.load_data()
    x_train = mnist_data[0][0].reshape([60000, 784])
    t_train = mnist_data[0][1].reshape([60000, ])
    x_test = mnist_data[1][0].reshape([10000, 784])
    t_test = mnist_data[1][1].reshape([10000, ])
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


def main():
    x, t = get_data()
    network = init_network()
    accuracy_count = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_count += 1
    print('Accuracy: {}'.format(accuracy_count/len(x)))



if __name__ == '__main__':
    main()
