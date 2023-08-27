from sympy import *
import numpy as np


class Function:
    def __init__(self):
        self.x = Symbol('x')
        self.function = self.x ** 2 + 3*self.x + 1
        self.derivative = self.function.diff(self.x)
        # The line below is a diff() method. Syntax: expr.diff(variable)
        # We also have diff function. Syntax diff(expr, variable)

    def value(self, point):
        f = lambdify(self.x, self.function, 'numpy')
        # Why 'numpy'?
        return f(point)

    def deriv(self, point):
        f = lambdify(self.x, self.derivative, 'numpy')
        return f(point)
