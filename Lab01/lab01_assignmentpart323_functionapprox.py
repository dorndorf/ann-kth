import numpy as np
import matplotlib.pyplot as plt
import learning_rules

x = np.linspace(-5, 5, 21)
y = np.linspace(-5, 5, 21)
targets = []
patterns = []
X, Y = np.meshgrid(x, y)
Z = np.exp(-X * X * 0.1) * np.exp(-Y*Y*0.1) - 0.5

for xi in x:
    for yi in y:
        targets.append(np.exp(-xi * xi * 0.1) * np.exp(-yi*yi*0.1) - 0.5)
        patterns.append([xi, yi, 1.0])

patterns = np.array(patterns).transpose()
targets = np.array(targets)[np.newaxis, :]

orig_patterns = patterns
orig_targets = targets

indices = np.arange(patterns.shape[1])
np.random.shuffle(indices)
targets = targets[:, indices]
patterns = patterns[:, indices]

lr = 0.01
epochs = 2000
alpha = 0.9
num_hidden = 20
batch = False
loss_all = []

for i in range(5):
    W, V,  loss_nobatch = learning_rules.two_layer_backprop(patterns, targets, epochs, lr, alpha, num_hidden, loss_name="rmse")
    loss_all.append(loss_nobatch[-1])

print("{0:.3f} ({1:.3f})".format(np.mean(np.array(loss_all)), np.std(np.array(loss_all))))

plt.plot(np.arange(len(loss_nobatch)), loss_nobatch, label="delta_nobatch")
plt.legend()
plt.show()

ndata = orig_patterns.shape[1]
hin = np.matmul(W, orig_patterns)
hout = np.concatenate((learning_rules.phi_function(hin), np.ones(ndata)[np.newaxis, :]))
oin = np.matmul(V, hout)
out = learning_rules.phi_function(oin)

out_p = np.ones((len(x), len(y)))
i = 0
for xi in range(len(x)):
    for yi in range(len(y)):
        out_p[xi, yi] = orig_targets[0, i]
        i += 1

plt.contourf(x, y, out_p)

plt.show()
