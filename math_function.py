from sympy import *
import numpy as np
import math

# Dat ten bien co nghia
# Quy tac dat ten
# - in thuong
# - cach nhau boi dau _

# Quy tac viet code
# - Co gang be no thanh nhieu phan nhat co the

total_study_time = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
exam_result = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

class LossFunction:
    def __init__(self):
        # Dat ten a -> threshold cho co y nghia
        self.threshold = Symbol('threshold')
        self.function = 0
        j = 1 # j = 0

        for j in range(len(total_study_time)):
            study_time = total_study_time[j]
            result = exam_result[j]
            self.function = self.function + \
                (result - (1 / (1 + math.e ** (self.threshold - study_time)))) ** 2 # lam qua nhieu viec trong mot hang

        self.derivative = self.function.diff(self.threshold)

    def sigmoid(self, x):
        # return 1 / (1 + math.e ** (-x))
        # cai nay khong nen boi vi dung phep tinh / , ** se rat cham voi python
        
        # Implement sigmoid function using numpy: exp(-x) = e ** (-x)
        return 1 / (1 + np.exp(-x))

    def predict(self, study_time):
        prediction = self.sigmoid(study_time - self.threshold)
        return prediction

    def value(self, a): # What is a?
        f = lambdify(self.threshold, self.function, 'numpy')
        return f(a)

    def gradient(self, a):
        f = lambdify(self.threshold, self.derivative, 'numpy')
        return f(a)