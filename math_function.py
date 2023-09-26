import numpy as np
total_study_time = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                             2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
exam_result = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1, 1, 1, 1, 1])


class LogLoss():
    def __init__(self):
        self.threshold = 0
        self.coefficient = 0
    
    def value(self, threshold, coefficient):
        self.threshold = threshold
        self.coefficient = coefficient
        for i in range(len(exam_result)):
            result = exam_result[i]
            
