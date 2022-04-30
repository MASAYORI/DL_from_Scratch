import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return (x > 0).astype(np.int32)


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1)


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)                   # Overflow prevention
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    plt.plot(x, y1, label="Step")
    plt.plot(x, y2, label="Sigmoid")
    plt.plot(x, y3, label="ReLU")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()