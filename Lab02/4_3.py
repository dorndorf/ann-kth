import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import learning_functions


data_train = np.genfromtxt('data_lab2/votes.dat', delimiter=',')
#with open('data_lab2/animalnames.txt', 'r') as file:
#    names = file.read().split()


x_train = data_train.reshape((349, 31))

num_nodes = 100

means = np.random.uniform(size=(num_nodes, x_train.shape[1]))
vars = np.ones(means.shape[0]) * 1.
weights = np.ones((means.shape[0], 1))

lr = 0.1
first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)

for i in range(20):
    num_neigh = int(6 - 6/19 * i)
    first_rbf.som_2d_neighbor_algorithm(x_train, num_neigh)

output = first_rbf.winning_index(x_train)

x_cord = np.mod(output, 10).astype(int)
y_cord = (output / 10).astype(int)



district = np.genfromtxt('data_lab2/mpdistrict.dat', delimiter='\n')
district_label = str(district)
party = np.genfromtxt('data_lab2/mpparty.dat', delimiter='\n')[2:]
party_names = ['none', 'm', 'fp', 's', 'v', 'mp', 'kd', 'c']
party_labels = [party_names[int(p)] for p in party]
party_colors = ['black', 'blue', 'yellow', 'red', 'darkred', 'lightgreen', 'black', 'darkgreen']
sex = np.genfromtxt('data_lab2/mpsex.dat', delimiter='\n')[1:]
sex_colors = ['blue', 'green']
sex_names = ['male', 'female']
sex_labels = [sex_names[int(p)] for p in sex]

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

x_cord = x_cord + np.random.normal(1.0, 0.14, size=len(x_cord))
y_cord = y_cord + np.random.normal(1.0, 0.14, size=len(y_cord))

for x, y, party_i, lab in zip(x_cord, y_cord, party, party_labels):
    col = party_colors[int(party_i)]
    ax1.scatter(x,
                y, c=col, label=lab)
ax1.set_xlim((0, 10.5))
ax1.set_ylim((0, 10.5))

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, label=par)
                   for col, par in zip(party_colors, party_names)]

ax1.legend(handles=legend_elements, prop={'size': 6})

for x, y, sex_i, lab in zip(x_cord, y_cord, sex, sex_labels):
    col = sex_colors[int(sex_i)]
    ax2.scatter(x,
                y, c=col, label=lab)
ax2.set_xlim((0, 10.5))
ax2.set_ylim((0, 10.5))

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, label=par)
                   for col, par in zip(sex_colors, sex_names)]

ax2.legend(handles=legend_elements, prop={'size': 6})

ax3.scatter(x_cord,
                y_cord, c=district, cmap='hsv')

ax3.set_xlim((0, 10.5))
ax3.set_ylim((0, 10.5))

ax1.set_title('Party')
ax2.set_title('Sex')
ax3.set_title('District')

plt.show()