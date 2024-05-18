import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(42)


MU = 6
SIGMA = 1


def p_x(x):
    return norm.pdf(x, MU, SIGMA)


class VariableTransform:
    def __init__(self, g, g_inv, p_x, p_y):
        self.g = g
        self.g_inv = g_inv
        self.p_x = p_x
        self.p_y = p_y

    def plot_graph(self, path, y_space):
        x = np.linspace(0, 10, 1000)
        y = np.linspace(y_space[0], y_space[1], 1000)
        x_transformed = self.g(y)

        # plot X PDF
        plt.plot(x, self.p_x(x), "r-", label="$p_x(x)$")
        plt.plot(self.p_y(y), y, "g-", label="$p_y(y)$")
        plt.plot(self.p_x(g(y)), y, "y-", label="$\\tilde{p}_y(y)$")

        # plot g(y)
        plt.plot(x, self.g_inv(x), "b-", label="$g^{-1}(x)$")

        # Plot maximum of p_x
        plt.scatter([6], [0], color="black", zorder=5)
        plt.scatter(6, self.g_inv(6), color="blue")
        plt.vlines([6], [0], self.g_inv(6), linestyle="dashed", color="red")
        plt.hlines(self.g_inv(6), 0, 6, linestyle="dashed", color="red")

        # plt.xlim((0, 10))
        # plt.ylim((0, 1))
        plt.legend()
        plt.savefig(path)
        plt.close()


# For non-linear transfer of variable
def g(y):
    return np.log(y) - np.log(1 - y) + 5


def g_inv(x):
    return 1 / (1 + np.exp(-x + 5))


def p_y(y):
    return p_x(g(y)) * np.abs((1 / y) + (1 / (1 - y)))


non_linear = VariableTransform(p_x=p_x, p_y=p_y, g=g, g_inv=g_inv)
non_linear.plot_graph("./images/exercise_1-4-non-linear.png", y_space=(0.001, 0.999))


# For linear transform of variable
def g(y):
    return 2 * (2 + y) - 3


def g_inv(x):
    return ((x + 3) / 2) - 2


def p_y(y):
    return p_x(g(y)) * np.abs(2)


linear = VariableTransform(p_x=p_x, p_y=p_y, g=g, g_inv=g_inv)
linear.plot_graph("./images/exercise_1-4-linear.png", y_space=(0.0, 4.0))
