import matplotlib.pyplot as plt
import numpy as np

# Plot the function


def plot_function_contour(f, x, y):
    # Plot the function using contour graph from matplotlib
    X, Y = np.meshgrid(x, y)
    Z = f.value(X, Y)
    plt.contourf(X, Y, Z, 100, cmap='RdGy')
    plt.colorbar()
    # plt.show()


def plot_changes(f, x, old_x, new_x, y, old_y, new_y, title='', show=False):
    # if old_x outside of x range, plot a point at the edge
    if old_x < x[0]:
        old_x = x[0]
    elif old_x > x[-1]:
        old_x = x[-1]
    old_z = f.value(old_x, old_y)

    if new_x < x[0]:
        new_x = x[0]
    elif new_x > x[-1]:
        new_x = x[-1]
    new_z = f.value(new_x, new_y)

    if old_y < y[0]:
        old_y = y[0]
    elif old_y > y[-1]:
        old_y = y[-1]
    old_z = f.value(old_x, old_y)

    if new_y < x[0]:
        new_y = x[0]
    elif new_y > y[-1]:
        new_y = y[-1]
    new_z = f.value(new_x, new_y)

    # plot the function
    z = f.value(x, y)
    X, Y = np.meshgrid(x, y)
    Z = f.value(X, Y)
    plt.contourf(X, Y, Z, 100, cmap='RdGy')
    plt.title(title)
    # Draw the old point as a red dot
    plt.plot(old_x, old_y, 'gray', marker='o', markersize=5)
    # Draw the new point as a red dot
    plt.plot(new_x, new_y, 'red', marker='o', markersize=5)
    # Draw a line between the old and new points
    plt.plot([old_x, new_x], [old_y, new_y], 'yellow')

    # Save the plot
    plt.savefig('./graphs/' + title + '.png')
    if (show):
        plt.show()