from sympy import *
import numpy as np
import math


data1 = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
data2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])


class SigmoidFunction:
    def __init__(self):
        self.a = Symbol('a')
        self.function = 0
        j = 1
        while (j < len(data1)):
            x = data1[j]
            y = data2[j]
            self.function = self.function + \
                (y - (1 / (1 + math.e ** (self.a - x)))) ** 2
            j += 1
        self.derivative = self.function.diff(self.a)

    def value(self, a):
        f = lambdify(self.a, self.function, 'numpy')
        return f(a)

    def gradient(self, a):
        f = lambdify(self.a, self.derivative, 'numpy')
        return f(a)
