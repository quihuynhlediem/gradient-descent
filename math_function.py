from sympy import Number
import numpy as np
import csv
with open("data.csv", 'r') as custfile:
    rows = csv.reader(custfile, delimiter=',')
    data = []
    for r in rows:
        data.append(r)


class LinearRegression:
    def __init__(self, v, b):
        self.v = v
        self.b = b
        j = 1
        self.function = 0
        while (j < len(data)):
            x = Number(data[j][0])
            y = Number(data[j][1])
            self.function = self.function + (y - self.v * x - self.b) ** 2
            self.derivative_v = 2 * self.v * x ** 2 - 2 * x * y + 2 * self.b * x
            self.derivative_b = 2 * self.b - 2 * y + 2 * self.v * x
            j += 1

    def value(self):
        return self.function

    def gradient(self):
        return np.array([self.derivative_v, self.derivative_b])
