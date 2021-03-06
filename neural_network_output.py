import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle

from activation_function import *


def get_data(normalize=True, one_hot_label=False):
    mnist_data = mnist.load_data()

    x_train = mnist_data[0][0].reshape([60000, 784])
    t_train = mnist_data[0][1].reshape([60000, ])
    x_test = mnist_data[1][0].reshape([10000, 784])
    t_test = mnist_data[1][1].reshape([10000, ])

    if one_hot_label:
        t_train = pd.get_dummies(t_train).values
        t_test = pd.get_dummies(t_test).values

    if normalize:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    return (x_train, t_train), (x_test, t_test)


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
    (_, _), (x, t) = get_data()
    network = init_network()
    batch_size = 100
    accuracy_count = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_count += np.sum(p == t[i:i+batch_size])
    print('Accuracy: {}'.format(accuracy_count/len(x)))


if __name__ == '__main__':
    main()
