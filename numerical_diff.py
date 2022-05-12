import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()

    return grad


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def function_tmp1(x0):
    return x0*x0 + 4.0**2.0


def tangent_line(f, x, t):
    """
    接線を求める関数
    :param f: 接線を求めたい関数
    :param x: 定義域
    :param t: 接点のx座標
    :return: np.ndarray型
    """
    return numerical_diff(f, t)*(x - t) + function_1(t)


def main():
    x0 = np.arange(-2.5, 2.5, 0.25)
    x1 = np.arange(-2.5, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]).T).T
    # line = tangent_line(function_2, x, 5)
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy")
    plt.grid()
    plt.draw()
    plt.show()

if __name__ == '__main__':
    main()

