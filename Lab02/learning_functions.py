import numpy as np


def phi_gaussian(x, my, sigma):
    return np.exp(-np.square(x-my) / (2*np.square(sigma)))


class GaussianRBF:

    def __init__(self, means, vars, weights, lr=0.01, som_lr=0.2):
        self.n = len(weights)
        self.means = means
        self.vars = vars
        self.weights = weights
        self.lr = lr
        self.som_lr = som_lr

    def set_learning_rate(self, lr, som_lr=0.2):
        self.lr = lr
        self.som_lr = som_lr

    def least_squares(self, x, y):
        phi_matrix = np.empty((len(x), self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])

        #rhs = np.matmul(np.transpose(phi_matrix), y)
        #lhs = np.matmul(np.transpose(phi_matrix), phi_matrix)
        w = np.matmul(np.linalg.pinv(phi_matrix), y)
        self.weights = w

    def delta_rule(self, x, y):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        for i, xk in enumerate(x):
            phi_k = phi_gaussian(xk, self.means, self.vars)
            err = y[i] - np.matmul(np.transpose(phi_k), self.weights)
            delta_w = self.lr * err * phi_k
            self.weights += delta_w

    def forward_pass(self, x):
        phi_matrix = np.empty((len(x), self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])
        return np.matmul(phi_matrix, self.weights)

    def res_error(self, x, y):
        phi_matrix = np.empty((len(x), self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])
        y_pred = np.matmul(phi_matrix, self.weights)
        return np.linalg.norm(y_pred - y, 2)

    def abs_res_error(self, x, y):
        phi_matrix = np.empty((len(x), self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])
        y_pred = np.matmul(phi_matrix, self.weights)
        return np.sum(np.abs(y_pred - y))

    def som_algorithm(self, x):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        for i, xk in enumerate(x):
            dk = np.abs(xk-self.means)
            min_index = np.argmin(dk)
            self.means[min_index] += self.som_lr * (xk-self.means)[min_index]



