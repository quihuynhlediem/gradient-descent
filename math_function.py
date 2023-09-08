from sympy import Number
import numpy as np
import csv
with open("data.csv", 'r') as custfile:
    rows = csv.reader(custfile, delimiter=',')
    data = []
    for r in rows:
        data.append(r)

j = 1
x_sum = 0
y_sum = 0
x2_sum = 0
y2_sum = 0
xy_sum = 0
while (j < len(data)):
    x = Number(data[j][0])
    y = Number(data[j][1])
    x_sum += x
    y_sum += y
    xy_sum += x * y
    x2_sum += x ** 2
    y2_sum += y ** 2
    j += 1

# print(x_sum)
# print(y_sum)
# print(x2_sum)
# print(y2_sum)
# print(xy_sum)

class LinearRegression:
    def __init__(self):
        self.v = 0
        self.b = 0
        # while (j < len(data)):
        #     x = Number(data[j][0])
        #     y = Number(data[j][1])
        #     self.function = self.function + (y - self.v * x - self.b) ** 2
        #     self.derivative_v = 2 * self.v * x ** 2 - 2 * x * y + 2 * self.b * x
        #     self.derivative_b = 2 * self.b - 2 * y + 2 * self.v * x
        #     j += 1

    def value(self, v, b):
        self.v = v
        self.b = b
        self.function = y2_sum + self.v ** 2 * x2_sum + self.b ** 2 - 2 * self.v * xy_sum - 2 * self.b * y_sum + 2 * self.v * self.b * x_sum
        return self.function

    def gradient(self, v, b):
        self.v = v
        self.b = b
        self.derivative_v = 2 * self.v * x2_sum - 2 * xy_sum + 2 * self.b * x_sum
        self.derivative_b = 2 * self.b - 2 * y_sum + 2 * self.v * x_sum
        return np.array([self.derivative_v, self.derivative_b])
