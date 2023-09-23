import numpy as np
total_study_time = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                             2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
exam_result = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1, 1, 1, 1, 1])


class LossFunction():
    def __init__(self):
        self.threshold = 1
        self.additional_variable = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, study_time):
        return self.sigmoid(self.additional_variable * study_time - self.threshold)

    def value(self, threshold, additional_variable):
        self.additional_variable = additional_variable
        self.threshold = threshold
        self.function = 0
        for i in range(len(total_study_time)):
            result = exam_result[i]
            study_time = total_study_time[i]
            self.function += (result - self.predict(study_time)) ** 2
        return self.function

    def deriv(self, threshold, additional_variable):
        self.additional_variable = additional_variable
        self.threshold = threshold
        self.derivative_threshold = 0
        self.derivative_additional_variable = 0
        for i in range(len(total_study_time)):
            result = exam_result[i]
            study_time = total_study_time[i]
            self.derivative_threshold += 2 * \
                (result - self.predict(study_time)) * \
                self.predict(study_time) * \
                (1 - self.predict(study_time))

            self.derivative_additional_variable += -2 * \
                (result - self.predict(study_time)) * \
                self.predict(study_time) * \
                (1 - self.predict(study_time)) * study_time
        return np.array([self.derivative_additional_variable, self.derivative_threshold])
