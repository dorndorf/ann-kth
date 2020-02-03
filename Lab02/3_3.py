import numpy as np
import matplotlib.pyplot as plt
import learning_functions

x_train = np.arange(0, 2*np.pi, 0.1)
y_train_sin = np.sin(2*x_train)
y_train_square = np.where(y_train_sin >= 0.0, 1, -1)

x_test = np.arange(0.05, 2*np.pi, 0.1)
y_test_sin = np.sin(2*x_test)
y_test_square = np.where(y_test_sin >= 0.0, 1, -1)

means = np.arange(0, 2*np.pi, 0.2)
vars = np.ones(len(means)) * 0.3
weights = np.zeros(len(means))
lr = 0.01
first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

first_rbf.som_algorithm(x_train)

first_rbf.least_squares(x_train, y_train_sin)

output = first_rbf.forward_pass(x_test)

plt.plot(x_test, y_test_sin)
plt.plot(x_test, output)
plt.show()

print(first_rbf.abs_res_error(x_test, y_test_sin))




