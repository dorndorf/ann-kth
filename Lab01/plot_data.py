import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

def plot_borderlines(patterns , targets, W, loss):
    indicesA = np.nonzero(targets[0] == 1)
    indicesB = np.nonzero(targets[0] != 1)
    classA = patterns[:2, indicesA]
    classB = patterns[:2, indicesB]

    #plt.close
    plt.scatter(classA[0], classA[1])
    plt.scatter(classB[0], classB[1])

    # border lines
    x = null_space(W[:, :-1])
    xh = np.linspace(-1, 1)
    yh = xh * x[1] / x[0]
    plt.plot(xh, yh + (W[0, 2] / np.linalg.norm(W)))
    plt.title('loss: {}'.format(loss[-1]))

    plt.ylim(-3,3)
    plt.xlim(-3,3)

    plt.show()
    plt.clf()
    return