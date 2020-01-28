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

#indices = np.arange(patterns.shape[1])
#np.random.shuffle(indices)
#targets = targets[:, indices]
#patterns = patterns[:, indices]

lr = 0.01
epochs = 250
alpha = 0.9
num_hidden = 20
batch = False
W, V, loss_nobatch = learning_rules.two_layer_backprop(patterns, targets, epochs, lr, alpha, num_hidden)

plt.plot(np.arange(len(loss_nobatch)), loss_nobatch, label="delta_nobatch")
plt.legend()
plt.show()

decision_pattern = np.zeros((3, 400))
meshtemp = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1, 1, 20))
decision_pattern[0, :] = np.array(meshtemp[0]).flatten()
decision_pattern[1, :] = np.array(meshtemp[1]).flatten()
decision_pattern[2, :] = np.ones(400)

ndata = decision_pattern.shape[1]
hin = np.matmul(W, decision_pattern)
hout = np.concatenate((learning_rules.phi_function(hin), np.ones(ndata)[np.newaxis, :]))
oin = np.matmul(V, hout)
result = learning_rules.phi_function(oin)

plt.contourf(meshtemp[0], meshtemp[1], np.sign(result).reshape(20,20), alpha=0.5)
plt.scatter(classA[0], classA[1])
plt.scatter(classB[0], classB[1])


#print(result)

plt.show()
