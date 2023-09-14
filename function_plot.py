import matplotlib.pyplot as plt
import numpy as np
from math_function import SigmoidFunction


def plot_function(f, x):
    y = f.value(x)
    plt.plot(x, y, color = "blue")
    plt.show()


def plot_changes(f, x, old_x, new_x, title='', show=False):
    # if old_x outside of x range, plot a point at the edge

    if new_x < x[0]:
        new_x = x[0]
    elif new_x > x[-1]:
        new_x = x[-1]
    new_y = f.value(new_x)

    if old_x < x[0]:
        old_x = x[0]
    elif old_x > x[-1]:
        old_x = x[-1]
    old_y = f.value(old_x)

    plt.title(title)

    plt.plot(old_x, old_y, 'gray', marker='o', markersize=5)

    plt.plot(new_x, new_y, 'red', marker='o', markersize=5)

    plt.plot([old_x, new_x], [old_y, new_y], 'yellow')

    y = f.value(x)
    plt.plot(x, y, color="blue")

    plt.savefig('./graphs/' + title + '.png')
    if (show):
        plt.show()
