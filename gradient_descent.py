import matplotlib.pyplot as plt
import numpy as np
from math_function import TwoVariables
from function_plot import plot_function_contour, plot_changes
from utils import clear_graph_folder, create_gifs
import math


def gradient_descent_process(starting_point_x, starting_point_y, learning_rate, precision, max_iterations):
    clear_graph_folder()

    x_old = starting_point_x
    y_old = starting_point_y
    # Initialize the counter: i
    i = 0
    #  Create a TwoVariables object: f
    f = TwoVariables()
    # Use linspace to sample 100 x, y values between -10 and 10
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    # Plot the function
    plot_function_contour(f, x, y)

    # Iterate until the change in x is less than precision or we hit the maximum number of iterations
    while True:
        if (x_old < x[0]): #limit the value of gradient
            break
        elif (x_old > x[-1]):
            break
        if (y_old < y[0]):
            break
        elif (y_old > y[-1]):
            break
        # Print out the number of the current iteration
        print("Iteration:", i)
        # Print out the old x value
        print("x_old:", x_old)
        # Print out the old y value
        print("y_old:", y_old)
        # Get the gradient at our current position
        x_gradient = f.deriv(x_old, y_old)[0]
        y_gradient = f.deriv(x_old, y_old)[1]
        # Print out the gradient
        print("x_gradient:", x_gradient)
        print("y_gradient:", y_gradient)
        # Move x_old by the negative of the gradient times the learning rate
        x_new = x_old - (x_gradient * learning_rate)  # Gradient Descent
        y_new = y_old - (y_gradient * learning_rate)  # Gradient Descent
        # Print out the new x,y value
        print("x_new:", x_new)
        print("y_new:", y_new)
        print("z: ", f.value(x_new, y_new))
        print('---------------')
        # Plot the function and new x value
        plot_changes(f, x, x_old, x_new, y, y_old, y_new, "" + str(i))
        # Check if the difference between the old x and new x is less than precision
        if math.sqrt((x_new - x_old)**2 + (y_new - y_old)**2) < precision:
            print("Precision reached!")
            break
        # Check if we've exceeded the maximum number of iterations
        if i > max_iterations:
            print("Maximum iterations exceeded!")
            break
        # Set the old x to the new x, old y to the new y
        x_old = x_new
        y_old = y_new
        # Increment the iteration counter
        i += 1

    create_gifs()


gradient_descent_process(1, 2, 0.4, 0.0000001, 100)
