import matplotlib.pyplot as plt
import numpy as np
from math_function import LogLoss
from function_plot import plot_function_contour, plot_changes
from utils import clear_graph_folder, create_gifs
import math


def gradient_descent_process(starting_point_coefficient, starting_point_threshold, learning_rate, precision, max_iterations):
    clear_graph_folder()

    coefficient_old = starting_point_coefficient
    threshold_old = starting_point_threshold

    i = 0

    f = LogLoss()

    coefficient = np.linspace(-8, 6, 150)
    threshold = np.linspace(-8, 6, 150)

    plot_function_contour(f, coefficient, threshold)

    while True:
        # if (threshold_old < threshold[0]): #limit the value of gradient
        #     break
        # elif (threshold_old > threshold[-1]):
        #     break
        # if (coefficient_old < coefficient[0]):
        #     break
        # elif (coefficient_old > coefficient[-1]):
        #     break

        print("Iteration:", i)

        print("coefficient_old:", coefficient_old)
        print("threshold_old:", threshold_old)

        coefficient_gradient = f.gradient(coefficient_old, threshold_old)[0]
        threshold_gradient = f.gradient(coefficient_old, threshold_old)[1]

        print("coefficient_gradient:", coefficient_gradient)
        print("threshold_gradient:", threshold_gradient)

        threshold_new = threshold_old - (threshold_gradient * learning_rate)
        coefficient_new = coefficient_old - \
            (coefficient_gradient * learning_rate)

        print("coefficient_new:", coefficient_new)
        print("threshold_new:", threshold_new)
        print("lossfunction: ", f.value(coefficient_new, threshold_new))
        print('---------------')

        plot_changes(f, coefficient, coefficient_old, coefficient_new,
                     threshold, threshold_old, threshold_new, "" + str(i))

        if math.sqrt((threshold_new - threshold_old)**2 + (coefficient_new - coefficient_old)**2) < precision:
            print("Precision reached!")
            break

        if i > max_iterations/2:
            learning_rate = learning_rate / 2

        if i > max_iterations:
            print("Maximum iterations exceeded!")
            break

        coefficient_old = coefficient_new
        threshold_old = threshold_new

        i += 1

    create_gifs()


gradient_descent_process(2, 2, 0.1, 0.0000001, 100)
