import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import learning_functions


x_train = np.array([[0.4000, 0.4439],
                [0.2439, 0.1463],
                [0.1707, 0.2293],
                [0.2293, 0.7610],
                [0.5171, 0.9414],
                [0.8732, 0.6536],
                [0.6878, 0.5219],
                [0.8488, 0.3609],
                [0.6683, 0.2536],
                [0.6195, 0.2634]])


num_nodes = 10

means = np.random.uniform(size=(num_nodes, x_train.shape[1]))
vars = np.ones(means.shape[0]) * 1.
weights = np.ones((means.shape[0], 1))

lr = 0.1
first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

for i in range(30):
    num_neigh = int(3 - 3/29 * i)
    first_rbf.som_circle_neighbor_algorithm(x_train, num_neigh)

output = first_rbf.winning_index(x_train)

print(output)

sort_index = np.argsort(output)

sorted_coords = x_train[sort_index]

for co in range(sorted_coords.shape[0]-1):
    plt.plot(sorted_coords[co:co+2, 0], sorted_coords[co:co+2, 1], 'ro-')

plt.plot([sorted_coords[-1, 0], sorted_coords[0, 0]], [sorted_coords[-1, 1], sorted_coords[0, 1]], 'ro-')

plt.show()