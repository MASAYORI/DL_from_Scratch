import numpy as np

import neural_network_output


def sum_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    batch_size = y.shape[0]
    delta = 1e-10
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def main():
    (x_train, t_train), (x_test, t_test) = neural_network_output.get_data(one_hot_label=True)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(cross_entropy_error(x_batch, t_batch))


if __name__ == '__main__':
    main()