import matplotlib.pyplot as plt
import numpy as np
from math_function import Function

# Plot the function


def plot_function(f, x):
    y = f.value(x)
    plt.plot(x, y)
    # Chấm toạ độ của x và y
    plt.show()


def plot_changes(f, x, old_x, new_x, title, show=False):
    # if old_x outside of x range, plot a point at the edge
    if old_x < x[0]:
        old_[x] = x[0]
    elif old_x > x[-1]:
        old_x = x[-1]
    old_y = f.value(old_x)

    if new_x < x[0]:
        new_x = x[0]
    elif new_x > x[-1]:
        new_x = x[-1]
    new_y = f.value(new_x)

    # plot the function
    y = f.value(x)
    plt.plot(x, y, color='black')
    # Có color hay không cũng không khác biết
    plt.title(title)
    # Draw the old point as gray dot
    plt.plot(old_x, old_y, 'gray', marker='o', markersize=5)
    # Draw the new point as red dot
    plt.plot(new_x, new_y, 'red', marker='o', markersize=5)
    # Draw a line between the old and new points
    plt.plot([old_x, new_x], [old_y, new_y], 'yellow')

    # Save the plot
    plt.savefig('./graphs/' + title + '.png')
    if (show):
        plt.show()
