import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import learning_functions

data_train = np.loadtxt('data_lab2/ballist.dat')
data_test = np.loadtxt('data_lab2/ballist.dat')

x_train = data_train[:, :2]
y_train = data_train[:, 2:]

x_test = data_test[:, :2]
y_test = data_test[:, 2:]

num_nodes = 20

means = np.empty((num_nodes, x_test.shape[1]))
means[:, 0] = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), num_nodes)
means[:, 1] = np.linspace(np.min(x_test[:, 1]), np.max(x_test[:, 1]), num_nodes)
vars = np.ones(means.shape[0]) * 0.3
weights = np.zeros((means.shape[0], y_test.shape[1]))
lr = 0.01
first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

first_rbf.som_algorithm(x_train)

first_rbf.least_squares(x_train, y_train)

output = first_rbf.forward_pass(x_test)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_test[:, 0], x_test[:, 1], y_test[:, 1], cmap='binary')
ax.scatter(x_test[:, 0], x_test[:, 1], output[:, 1], cmap='binary')

plt.show()

print(first_rbf.abs_res_error(x_test, y_test))

