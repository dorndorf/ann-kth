import numpy as np
import matplotlib.pyplot as plt
import learning_functions

mean_distance = np.linspace(0.5,0.05,200)
for i in range(len(mean_distance)):
    x_train = np.arange(0, 2*np.pi, 0.1)[:, np.newaxis]
    y_train_sin = np.sin(2*x_train)
    y_train_square = np.where(y_train_sin >= 0.0, 1, -1).astype(float)

    x_test = np.arange(0.05, 2*np.pi, 0.1)[:, np.newaxis]
    y_test_sin = np.sin(2*x_test)
    y_test_square = np.where(y_test_sin >= 0.0, 1, -1).astype(float)

    means = np.arange(0, 2*np.pi, mean_distance[i])[:, np.newaxis]
    vars = np.ones(means.shape[0]) * 0.1

    abs_res_error_all = []

    #for i in range(5):
    weights = np.zeros(means.shape[0])[:, np.newaxis]


    first_rbf = learning_functions.GaussianRBF(means, vars, weights)

    first_rbf.least_squares(x_train, y_train_sin)

    output = first_rbf.forward_pass(x_test)

    abs_res_error_all.append(first_rbf.abs_res_error(x_test, y_test_sin))

    #plt.plot(x_test, y_test_square)
    #plt.plot(x_test, output)
    #plt.show()

    print("{0:.4f} ({1:.4f})".format(np.mean(np.array(abs_res_error_all)), np.std(np.array(abs_res_error_all))))
    print("Number of nodes: {}".format(first_rbf.n))




