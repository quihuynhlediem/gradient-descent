import numpy as np
from math_function import Function
from function_plot import plot_function, plot_changes
from utils import clear_graph_folder, create_gifs


def gradient_descent_process(starting_point, learning_rate, precision, max_iterations):
    clear_graph_folder()

    x_old = starting_point
    # Initialize the counter: i
    i = 0
    #  Create a Function object: f
    f = Function()
    # Use linspace to sample 100 x values between -10 and 10
    x = np.linspace(-5, 5, 100)
    # Plot the function
    plot_function(f, x)

    # Iterate until the change in x is less than precision or we hit the maximum number of iterations
    while True:
        # Print out the number of the current iteration
        print("Iteration:", i)
        # Print out the old x value
        print("x_old:", x_old)
        # Get the gradient at our current position
        gradient = f.deriv(x_old)
        # Print out the gradient
        print("gradient:", gradient)
        # Move x_old by the negative of the gradient times the learning rate
        x_new = x_old - (gradient * learning_rate)  # Gradient Descent
        # Print out the new x value
        print("x_new:", x_new)
        # Plot the function and new x value
        plot_changes(f, x, x_old, x_new, "" + str(i))
        # Check if the difference between the old x and new x is less than precision
        if abs(x_new - x_old) < precision:
            print("Precision reached!")
            break
        # Check if we've exceeded the maximum number of iterations
        if i > max_iterations:
            print("Maximum iterations exceeded!")
            brek
        # Set the old x to the new x
        x_old = x_new
        # Increment the iteration counter
        i += 1

    create_gifs()


gradient_descent_process(3, 0.2, 0.000001, 100)
