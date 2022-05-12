import functools

from layer_naive import *


def main():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_total_layer = AddLayer()
    mul_price_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    total_price = add_total_layer.forward(apple_price, orange_price)
    price = mul_price_layer.forward(total_price, tax)

    dprice = 1
    dtotal_price, dtax = mul_price_layer.backward(dprice)
    dapple_price, dorange_price = add_total_layer.backward(dtotal_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)

    d_list = list(map(functools.partial(round, ndigits=1), [dapple, dapple_num, dorange, dorange_num, dtax]))
    print(*d_list)

    print(round(price))


if __name__ == '__main__':
    main()