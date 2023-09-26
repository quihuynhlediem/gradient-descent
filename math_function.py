import numpy as np
total_study_time = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                             2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
exam_result = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1, 1, 1, 1, 1])


class LogLoss():
    def __init__(self):
        self.threshold = 0
        self.coefficient = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, study_time):
        log_odds = self.coefficient * study_time + self.threshold
        return self.sigmoid(log_odds)

    def value(self, coefficient, threshold):
        self.coefficient = coefficient
        self.threshold = threshold
        self.loss_function = 0
        for i in range(len(exam_result)):
            result = exam_result[i]
            study_time = total_study_time[i]
            self.loss_function += -(result * np.log(self.predict(study_time)) +
                                    (1 - result) * np.log(1 - self.predict(study_time)))
        return self.loss_function

    def gradient(self, coefficient, threshold):
        self.coefficient = coefficient
        self.threshold = threshold
        self.threshold_gradient = 0
        self.coefficient_gradient = 0
        for i in range(len(exam_result)):
            result = exam_result[i]
            study_time = total_study_time[i]
            self.threshold_gradient += -(result - self.predict(study_time))
            self.coefficient_gradient += -(result - self.predict(study_time)) * study_time
        return np.array([self.coefficient_gradient, self.threshold_gradient])
