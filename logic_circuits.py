import numpy as np


def logic_and(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.dot(w.T, x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def logic_nand(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.dot(w.T, x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def logic_or(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.dot(w.T, x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def logic_xor(x1, x2):
    s1 = logic_nand(x1, x2)
    s2 = logic_or(x1, x2)
    return logic_and(s1, s2)


def main():
    input_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (i, j) in input_list:
        print('AND: {} => {}'.format((i, j), logic_and(i, j)))
    print('-'*20)
    for (i, j) in input_list:
        print('NAND: {} => {}'.format((i, j), logic_nand(i, j)))
    print('-'*20)
    for (i, j) in input_list:
        print('OR: {} => {}'.format((i, j), logic_or(i, j)))
    print('-'*20)
    for (i, j) in input_list:
        print('XOR: {} => {}'.format((i, j), logic_xor(i, j)))


if __name__ == '__main__':
    main()