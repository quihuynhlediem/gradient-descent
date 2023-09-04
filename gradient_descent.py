import matplotlib.pyplot as plt
import numpy as np
from math_function import LinearRegression
from function_plot import plot_function_contour, plot_changes
from utils import clear_graph_folder, create_gifs
import math


def gradient_descent_process(starting_point_v, starting_point_b, learning_rate, precision, max_iterations):
    clear_graph_folder()

    v_old = starting_point_v
    b_old = starting_point_b

    i = 0

    f = LinearRegression(v_old, b_old)

    v = np.linspace(30, 40, 100)
    b = np.linspace(10, 20, 100)

    # plot_function_contour(v, b)

    while True:
        if (v_old < v[0]):  # limit the value of gradient
            break
        elif (v_old > v[-1]):
            break
        if (b_old < b[0]):
            break
        elif (b_old > b[-1]):
            break

        print("Iteration:", i)
        print("v_old:", v_old)
        print("b_old:", b_old)

        v_gradient = f.gradient()[0]
        b_gradient = f.gradient()[1]

        print("x_gradient:", v_gradient)
        print("y_gradient:", b_gradient)

        v_new = v_old - (v_gradient * learning_rate)
        b_new = b_old - (b_gradient * learning_rate)

        print("v_new:", v_new)
        print("b_new:", b_new)
        f = LinearRegression(v_new, b_new)
        print("z: ", f.value())
        print('---------------')

        # plot_changes(f, v, v_old, v_new, b, b_old, b_new, '' + str(i))
        if math.sqrt((v_new - v_old)**2 + (b_new - b_old)**2) < precision:
            print("Precision reached!")
            break
        if i > max_iterations:
            print("Maximum iterations exceeded!")
            break

        v_old = v_new
        b_old = b_new
        i += 1

    # create_gifs()


gradient_descent_process(35, 15, 0.005, 0.0000001, 200)
