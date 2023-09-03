from sympy import *
import numpy as np


class Function:
    def __init__(self):
        self.x = Symbol('x')
        self.function = self.x ** 2 + sin(5 * self.x) + 1
        self.derivative = self.function.diff(self.x)

    # Calculate f(point)
    def value(self, point):
        f = lambdify(self.x, self.function, 'numpy')
        return f(point)

    # Calculate f'(point)
    def deriv(self, point):
        f = lambdify(self.x, self.derivative, 'numpy')
        return f(point)


class TwoVariables:
    def __init__(self):
        self.x = Symbol('x')
        self.y = Symbol('y')
        self.function = self.x ** 2 * self.y + sin(self.y) + 1
        #self.function = self.x ** 2 * self.y ** 2 + 1
        self.derivative_x = self.function.diff(self.x)
        self.derivative_y = self.function.diff(self.y)
        # The line below is a diff() method. Syntax: expr.diff(variable)
        # We also have diff function. Syntax diff(expr, variable)

    def value(self, x, y):
        f = lambdify([self.x, self.y], self.function, 'numpy')
        # Why 'numpy'?
        return f(x, y)

    def deriv(self, x, y):
        f_x = lambdify([self.x, self.y], self.derivative_x, 'numpy')
        f_y = lambdify([self.x, self.y], self.derivative_y, 'numpy')
        return np.array([f_x(x, y), f_y(x, y)])