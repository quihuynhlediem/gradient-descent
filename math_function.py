from sympy import *
import numpy as np
import csv
with open("data.csv", 'r') as custfile:
    rows = csv.reader(custfile, delimiter=',')
    data = []
    for r in rows:
        data.append(r)

# j = 1
# x_sum = 0
# y_sum = 0
# x2_sum = 0
# y2_sum = 0
# xy_sum = 0
# while (j < len(data)):
#     x = Number(data[j][0])
#     y = Number(data[j][1])
#     x_sum = x_sum + x
#     y_sum = y_sum + y
#     xy_sum = xy_sum + x * y
#     x2_sum = x2_sum + x ** 2
#     y2_sum = y2_sum + y ** 2
#     j += 1

# print("x_sum:", x_sum)
# print("y_sum", y_sum)
# print("x2_sum:", x2_sum)
# print("y2_sum", y2_sum)
# print("xy_sum", xy_sum)


class LinearRegression:
    def __init__(self):
        self.v = Symbol('v')
        self.b = Symbol('b')
        self.function = 0
        j = 1
        while (j < len(data)):
            x = Number(data[j][0])
            y = Number(data[j][1])
            self.function = self.function + (self.v * x + self.b - y) ** 2
            j += 1
        # self.function = x2_sum * self.v ** 2 + self.b ** 2 + y2_sum + 2 * self.v * self.b * x_sum - 2 * self.v * xy_sum - 2 * self.b * y_sum
        self.derivative_v = self.function.diff(self.v)
        self.derivative_b = self.function.diff(self.b)
        # self.derivative_v = 2 *self.v * x2_sum - 2 * xy_sum + 2 * self.b * x_sum
        # self.derivative_b = 2 * self.b - 2 * y_sum + 2 * self.v * x_sum 

    def value(self, v, b):
        f = lambdify([self.v, self.b], self.function, 'numpy')
        return f(v, b)

    def gradient(self, v, b):
        f_v = lambdify([self.v, self.b], self.derivative_v, 'numpy')
        f_b = lambdify([self.v, self.b], self.derivative_b, 'numpy')
        return np.array([f_v(v, b), f_b(v, b)])
