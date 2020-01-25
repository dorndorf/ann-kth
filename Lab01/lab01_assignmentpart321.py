import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import learning_rules

n = 100
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3
classA = np.zeros((3, n))
classA[0, :50] = np.random.normal(-mA[0], sigmaA, 50)
classA[0, 50:] = np.random.normal(mA[0], sigmaA, 50)
classA[1, :] = np.random.normal(mA[1], sigmaA, n)
classA[2, :] = np.ones(n)
classB = np.zeros((3, n))
classB[0, :] = np.random.normal(mB[0], sigmaB, n)
classB[1, :] = np.random.normal(mB[1], sigmaB, n)
classB[2, :] = np.ones(n)

#plt.scatter(classA[0], classA[1])
#plt.scatter(classB[0], classB[1])
#plt.show()

patterns = np.concatenate((classA, classB), axis=1)
targets = np.ones(n*2)[np.newaxis, :]
targets[0, n:] = -1

indices = np.arange(patterns.shape[1])
np.random.shuffle(indices)
targets = targets[:, indices]
patterns = patterns[:, indices]

lr = 0.001
epochs = 20
alpha = 0.9
num_hidden = 1000
batch = False
W, loss_nobatch = learning_rules.two_layer_backprop(patterns, targets, W, V, epochs, lr, alpha, num_hidden)

plt.plot(np.arange(len(loss_nobatch)), loss_nobatch, label="delta_nobatch")
plt.legend()

result = np.matmul(W, patterns)
#print(result)

plt.show()


## Perceptron

targets[targets == -1] = 0

W = np.random.normal(0.0, 0.1, size=(targets.shape[0], patterns.shape[0]))

lr = 0.001
epochs = 1

W, loss = learning_rules.perceptron_rule(patterns, targets, W, epochs, lr)

x = null_space(W[:, :-1])
xh = np.linspace(-1, 1)
yh = xh * x[1]/x[0]
plt.plot(xh, yh+(W[0, 2]/np.linalg.norm(W)))
#plt.show()