import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import learning_functions


data_train = np.genfromtxt('data_lab2/animals.dat', delimiter=',')
with open('data_lab2/animalnames.txt', 'r') as file:
    names = file.read().split()


x_train = data_train.reshape((32, 84))

num_nodes = 100

means = np.random.uniform(size=(num_nodes, x_train.shape[1]))
vars = np.ones(means.shape[0]) * 1.
weights = np.ones((means.shape[0], 1))

lr = 0.1
first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

min_max = []
uni = []

for neig in range(1, 50):
    mini = []
    unii = []
    for run in range(5):
        first_rbf = learning_functions.GaussianRBF(means, vars, weights)
        for i in range(20):
            num_neigh = int(neig - neig/19 * i)
            first_rbf.som_index_neighbor_algorithm(x_train, num_neigh)

        output = first_rbf.winning_index(x_train)
        mini.append(np.max(output) - np.min(output))
        unii.append(len(np.unique(output)))

    min_max.append(np.mean(mini))
    uni.append(np.mean(unii))

plt.plot(np.arange(len(min_max)), min_max, label='Range of Groups')
plt.plot(np.arange(len(min_max)), uni, label='# Different Groups')
plt.legend()
plt.xlabel('# Neighbours')
plt.show()


for itr in range(100):
    hit = np.where(output == itr)
    if len(hit[0]):
        print(itr, ": ", end="")
        for h in hit[0]:
            print(names[h], " ", end="")
        print("")
    else:
        continue


