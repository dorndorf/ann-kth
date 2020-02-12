import numpy as np
import matplotlib.pyplot as plt
import learning_functions

# training and testdata
x_train = np.arange(0, 2*np.pi, 0.1)[:, np.newaxis]
y_train_sin = np.sin(2*x_train)
y_train_sin += np.random.normal(0.0, 0.1, y_train_sin.shape[0])[:, np.newaxis]

x_test = np.arange(0.05, 2*np.pi, 0.1)[:, np.newaxis]
y_test_sin = np.sin(2*x_test)
y_test_sin += np.random.normal(0.0, 0.1, y_test_sin.shape[0])[:, np.newaxis]

mean_distance = 0.4
var_value = 0.2

# SOM: winner takes it all
means = np.arange(0, 2*np.pi, mean_distance)[:, np.newaxis]
vars = np.ones(means.shape[0]) * var_value
weights = np.zeros(means.shape[0])[:, np.newaxis]

first_rbf = learning_functions.GaussianRBF(means, vars, weights)

for i in range(20):
    first_rbf.som_algorithm(x_train)

first_rbf.least_squares(x_train, y_train_sin)
output_som = first_rbf.forward_pass(x_test)

# without SOM
means = np.arange(0, 2*np.pi, mean_distance)[:, np.newaxis]
vars = np.ones(means.shape[0]) * var_value
weights = np.zeros(means.shape[0])[:, np.newaxis]

second_rbf = learning_functions.GaussianRBF(means, vars, weights)

for i in range(20):
    second_rbf.least_squares(x_train, y_train_sin)

output = second_rbf.forward_pass(x_test)

# SOM: with neighborhood
means = np.arange(0, 2*np.pi, mean_distance)[:, np.newaxis]
vars = np.ones(means.shape[0]) * var_value
weights = np.zeros(means.shape[0])[:, np.newaxis]

third_rbf = learning_functions.GaussianRBF(means, vars, weights, 2)
third_rbf.som_index_neighbor_algorithm(x_train)
third_rbf.least_squares(x_train, y_train_sin)
output_som_neighbor = third_rbf.forward_pass(x_test)

# plotting and printing
plt.plot(x_test, y_test_sin, label = 'testdata')
plt.plot(x_test, output_som, label = 'SOM: Winner only')
plt.plot(x_test, output, label = 'without SOM')
plt.plot(x_test, output_som_neighbor, label = 'SOM: neighborhood')
plt.legend()
plt.title('Squared Residual Error with SOM and without Noise')
plt.show()
#plt.savefig('plots/ErrorSinSomWithoutNoise.png')

print('Squared Residual Error with SOM (winner takes it all): {0:.4f} ' .format(first_rbf.res_error(x_test, y_test_sin)))
print('Squared Residual Error without SOM: {0:.4f} ' .format(second_rbf.res_error(x_test, y_test_sin)))
print('Squared Residual Error with SOM (neighborhood): {0:.4f} ' .format(third_rbf.res_error(x_test, y_test_sin)))
print("Number of nodes: {}".format(first_rbf.n))
print("Width of nodes: {}".format(first_rbf.vars[1]))




