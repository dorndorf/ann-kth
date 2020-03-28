import numpy as np

def phi_gaussian(x, my, sigma):
    r = np.linalg.norm(x-my, 2, axis=-1)
    return np.exp(-np.square(r) / (2*np.square(sigma)))

class GaussianRBF:

    def __init__(self, means, vars, weights, lr=0.01, som_lr=0.2):
        self.n = weights.shape[0]  # weights shape is (number of nodes * output size)
        self.means = means  # means shape is (number of nodes * input size)
        self.vars = vars  # var shape is (number of nodes)
        self.weights = weights # weights shape is (number of nodes * output size)
        self.lr = lr
        self.som_lr = som_lr
        self.input_dim = means.shape[1]
        self.output_dim = weights.shape[1]

    def set_learning_rate(self, lr, som_lr=0.2):
        self.lr = lr
        self.som_lr = som_lr

    def least_squares(self, x, y):
        phi_matrix = np.empty((x.shape[0], self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])
        for dim_i in range(self.output_dim):
            w = np.matmul(np.linalg.pinv(phi_matrix), y[:, dim_i])
            self.weights[:, dim_i] = w

    def delta_rule(self, x, y):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        for i in range(x.shape[0]):
            phi_k = phi_gaussian(np.repeat(x[i], self.means.shape[0], axis=0).reshape(self.means.shape), self.means, self.vars)
            err = y[i] - np.matmul(np.transpose(phi_k), self.weights)
            delta_w = self.lr * err * phi_k
            self.weights += delta_w.reshape(self.weights.shape)

    def forward_pass(self, x):
        phi_matrix = np.empty((x.shape[0], self.n))
        for i in range(phi_matrix.shape[0]):
            for j in range(phi_matrix.shape[1]):
                phi_matrix[i, j] = phi_gaussian(x[i], self.means[j], self.vars[j])
        return np.matmul(phi_matrix, self.weights)

    def res_error(self, x, y):
        y_pred = self.forward_pass(x)
        return np.divide(np.linalg.norm(y_pred - y, 2),y.shape[0])

    def abs_res_error(self, x, y):
        y_pred = self.forward_pass(x)
        return np.divide(np.sum(np.abs(y_pred - y)),y.shape[0])

    def som_algorithm(self, x):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i]-self.means, 2, axis=1)
            min_index = np.argmin(dk)
            self.means[min_index, :] += self.som_lr * (x[i]-self.means[min_index])

    def som_neighbor_algorithm(self, x, num_neigh=2):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i]-self.means, 2, axis=1)
            min_index = np.argsort(dk)[:num_neigh]
            self.means[min_index, :] += self.som_lr * (
                    np.tile(x[i], num_neigh).reshape((-1, x.shape[1])) - self.means[min_index, :])

    def winning_index(self, x):
        output = []
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i] - self.means, 1, axis=1)
            min_index = np.argmin(dk)
            output.append(min_index)
        return np.array(output)

    def som_index_neighbor_algorithm(self, x, num_neigh=2):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        half_neigh = int(num_neigh/2)
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i]-self.means, 1, axis=1)
            min_index = np.argmin(dk)
            half_min = min_index - half_neigh
            half_max = min_index + half_neigh + 1
            if half_min < 0:
                half_min = int(0)
                half_max = min_index + (min_index - half_min)
            if half_max >= self.means.shape[0]:
                half_max = int(self.means.shape[0])
                half_min = min_index + (half_max - min_index)
            act_num_neigh = half_max - half_min
            self.means[half_min:half_max, :] += self.som_lr * (
                    np.tile(x[i], act_num_neigh).reshape((-1, x.shape[1])) - self.means[half_min:half_max, :])

    def som_circle_neighbor_algorithm(self, x, num_neigh=2):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        half_neigh = int(num_neigh/2)
        array_length = x.shape[0]
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i]-self.means, 1, axis=1)
            min_index = np.argmin(dk)
            half_min = min_index - half_neigh
            half_max = min_index + half_neigh + 1
            index_array = np.arange(half_min, half_max, 1).astype(int)
            for curr_index in index_array:
                if curr_index >= array_length:
                    temp_index = curr_index - array_length
                    self.means[temp_index, :] += self.som_lr * (
                            x[i] - self.means[temp_index, :])
                else:
                    self.means[curr_index, :] += self.som_lr * (
                            x[i] - self.means[curr_index, :])

    def som_2d_neighbor_algorithm(self, x, num_neigh=2):
        # shuffle data
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices, :]
        for i in range(x.shape[0]):
            dk = np.linalg.norm(x[i]-self.means, 1, axis=1)
            min_index = np.argmin(dk)
            x_cord = int(min_index % 10)
            y_cord = int(min_index / 10)
            x_neighbors = np.arange(x_cord - num_neigh, x_cord + num_neigh)
            y_neighbors = np.arange(y_cord - num_neigh, y_cord + num_neigh)
            neigh_mesh = np.meshgrid(x_neighbors[np.logical_and(x_neighbors >= 0, x_neighbors < 10)],
                                     y_neighbors[np.logical_and(y_neighbors >= 0, y_neighbors < 10)])
            index_array = np.unique(neigh_mesh[0] + neigh_mesh[1]*10).astype(int).flatten()
            if not len(index_array):
                self.means[min_index, :] += self.som_lr * (x[i] - self.means[min_index, :])
            for curr_index in index_array:
                self.means[curr_index, :] += self.som_lr * (x[i] - self.means[curr_index, :])
