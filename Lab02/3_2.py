import numpy as np
import matplotlib.pyplot as plt
import learning_functions

x_train = np.arange(0, 2*np.pi, 0.1)[:, np.newaxis]
y_train_sin = np.sin(2*x_train)
y_train_square = np.where(y_train_sin >= 0.0, 1, -1).astype(float)

y_train_sin += np.random.normal(0.0, 0.1, y_train_sin.shape[0])[:, np.newaxis]
y_train_square += np.random.normal(0.0, 0.1, y_train_square.shape[0])[:, np.newaxis]

x_test = np.arange(0.05, 2*np.pi, 0.1)[:, np.newaxis]
y_test_sin = np.sin(2*x_test)
y_test_square = np.where(y_test_sin >= 0.0, 1, -1).astype(float)

y_test_sin += np.random.normal(0.0, 0.1, y_test_sin.shape[0])[:, np.newaxis]
y_test_square += np.random.normal(0.0, 0.1, y_test_square.shape[0])[:, np.newaxis]

means = np.arange(0, 2*np.pi, 0.2)[:, np.newaxis]
vars = np.ones(means.shape[0]) * 0.1
weights = np.zeros(means.shape[0])[:, np.newaxis]

lr = 0.01
epochs = 100

first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

for ep in range(epochs):
    first_rbf.delta_rule(x_train, y_train_square)

output = first_rbf.forward_pass(x_test)

plt.plot(x_test, y_test_square)
plt.plot(x_test, output)
plt.show()

print(first_rbf.abs_res_error(x_test, y_test_square))




