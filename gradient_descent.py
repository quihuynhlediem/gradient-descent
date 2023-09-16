import matplotlib.pyplot as plt
import numpy as np
from math_function import SigmoidFunction, probability
from function_plot import plot_function, plot_changes
from utils import clear_graph_folder, create_gifs


def gradient_descent_process(starting_point, learning_rate, precision, max_iterations):
    clear_graph_folder()

    a_old = starting_point

    i = 0

    f = SigmoidFunction()

    a = np.linspace(-5, 5, 100)

    plot_function(f, a)

    while True:
        if (a_old < a[0]):
            break
        elif (a_old > a[-1]):
            break

        print("Iteration:", i)
        print("a_old:", a_old)

        gradient = f.gradient(a_old)

        print("gradient:", gradient)

        a_new = a_old - (gradient * learning_rate)

        print("a_new:", a_new)
        print("loss function: ", f.value(a_new))
        print('---------------')

        plot_changes(f, a, a_old, a_new, '' + str(i))
        if (a_new - a_old) < precision:
            print("Precision reached!")
            return a_new
            break
        if i > max_iterations:
            print("Maximum iterations exceeded!")
            return a_new
            break

        a_old = a_new
        i += 1

    create_gifs()


print(probability(1, gradient_descent_process(1, 0.05, 0.00001, 200)))
