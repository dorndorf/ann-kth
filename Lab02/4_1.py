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

for i in range(20):
    num_neigh = int(50 - 50/19 * i)
    first_rbf.som_index_neighbor_algorithm(x_train, num_neigh)


output = first_rbf.winning_index(x_train)

for itr in range(100):
    hit = np.where(output == itr)
    if len(hit[0]):
        for h in hit[0]:
            print(names[h], " ", end="")
    else:
        print(' ')


