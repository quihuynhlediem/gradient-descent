import matplotlib.pyplot as plt
import numpy as np
from math_function import LossFunction
from function_plot import plot_function_contour, plot_changes
from utils import clear_graph_folder, create_gifs
import math


def gradient_descent_process(starting_point_threshold, starting_point_additional_variable, learning_rate, precision, max_iterations):
    clear_graph_folder()

    threshold_old = starting_point_threshold
    additional_variable_old = starting_point_additional_variable

    i = 0

    f = LossFunction()

    threshold = np.linspace(1, 7, 100)
    additional_variable = np.linspace(0, 4, 100)

    plot_function_contour(f, threshold, additional_variable)


    while True:
        # if (threshold_old < threshold[0]): #limit the value of gradient
        #     break
        # elif (threshold_old > threshold[-1]):
        #     break
        # if (additional_variable_old < additional_variable[0]):
        #     break
        # elif (additional_variable_old > additional_variable[-1]):
        #     break

        print("Iteration:", i)

        print("threshold_old:", threshold_old)

        print("additional_variable_old:", additional_variable_old)

        threshold_gradient = f.deriv(threshold_old, additional_variable_old)[1]
        additional_variable_gradient = f.deriv(threshold_old, additional_variable_old)[0]

        print("threshold_gradient:", threshold_gradient)
        print("additional_variable_gradient:", additional_variable_gradient)

        threshold_new = threshold_old - (threshold_gradient * learning_rate)  
        additional_variable_new = additional_variable_old - (additional_variable_gradient * learning_rate)

        print("threshold_new:", threshold_new)
        print("additional_variable_new:", additional_variable_new)
        print("lossfunction: ", f.value(threshold_new, additional_variable_new))
        print('---------------')

        plot_changes(f, threshold, threshold_old, threshold_new, additional_variable, additional_variable_old, additional_variable_new, "" + str(i))

        if math.sqrt((threshold_new - threshold_old)**2 + (additional_variable_new - additional_variable_old)**2) < precision:
            print("Precision reached!")
            break

        if i > max_iterations:
            print("Maximum iterations exceeded!")
            break

        threshold_old = threshold_new
        additional_variable_old = additional_variable_new

        i += 1

    create_gifs()


gradient_descent_process(1, 1, 0.05, 0.0000001, 100)
