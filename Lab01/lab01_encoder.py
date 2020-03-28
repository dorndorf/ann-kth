import numpy as np
import matplotlib.pyplot as plt
import learning_rules

patterns = np.ones((9, 8))
patterns = patterns * -1
for i in range(8):
    patterns[i,i] = 1
patterns[-1, :] = 1

targets = patterns[:-1, :]

lr = 0.1
epochs = 2000
alpha = 0.9
num_hidden = 2
batch = False
W, V, loss_nobatch = learning_rules.two_layer_backprop(patterns, targets, epochs, lr, alpha, num_hidden)

plt.plot(np.arange(len(loss_nobatch)), loss_nobatch, label="delta_nobatch")
plt.legend()
plt.show()

ndata = patterns.shape[1]
hin = np.matmul(W, patterns)
hout = np.concatenate((learning_rules.phi_function(hin), np.ones(ndata)[np.newaxis, :]))
oin = np.matmul(V, hout)
result = learning_rules.phi_function(oin)

print(patterns)
print(np.sign(result))
