import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import learning_rules

n = 100
#mA = [1, 0.5]
mA = [1.0, 1.0]
sigmaA = 0.5
#mB = [-1, 0.0]
mB = [3.0, 3.0]
sigmaB = 0.5
classA = np.zeros((2, n))   # used for calculation without bias term
#classA = np.zeros((3, n))
classA[0, :] = np.random.normal(mA[0], sigmaA, n)
classA[1, :] = np.random.normal(mA[1], sigmaA, n)
#classA[2, :] = np.ones(n)
classB = np.zeros((2, n))   # used for calculation without bias term
#classB = np.zeros((3, n))
classB[0, :] = np.random.normal(mB[0], sigmaB, n)
classB[1, :] = np.random.normal(mB[1], sigmaB, n)
#classB[2, :] = np.ones(n)

# plot classes
plt.scatter(classA[0], classA[1])
plt.scatter(classB[0], classB[1])
plt.savefig('plots/3.1.3/classes.svg')
plt.clf()

patterns = np.concatenate((classA, classB), axis=1)
targets = np.ones(n * 2)[np.newaxis, :]
targets[0, n:] = -1

indices = np.arange(patterns.shape[1])
np.random.shuffle(indices)
targets = targets[:, indices]
patterns = patterns[:, indices]

lr = 0.005
epochs = 15
num_runs = 1

loss_perceptron_all = np.zeros((num_runs, epochs ))
loss_delta_batch_all = np.zeros((num_runs, epochs ))
loss_delta_nobatch_all = np.zeros((num_runs, epochs ))

for num in range(num_runs):
    targets[targets == 0] = -1      #reset

    batch = False
    W_delta_nobatch, loss_delta_nobatch = learning_rules.delta_rule(patterns, targets, epochs, lr, batch)

    batch = True
    W_delta_batch, loss_delta_batch = learning_rules.delta_rule(patterns, targets, epochs, lr, batch)

    targets[targets == -1] = 0
    batch = True
    animation = True
    W_perceptron, loss_perceptron = learning_rules.perceptron_rule(patterns, targets, epochs, lr, batch)

    loss_perceptron_all[num] = np.add(loss_perceptron_all[num],np.array(loss_perceptron))
    loss_delta_nobatch_all[num] = np.add(loss_delta_nobatch_all[num], np.array(loss_delta_nobatch))
    loss_delta_batch_all[num] = np.add(loss_delta_batch_all[num], np.array(loss_delta_batch))

    # classes
    plt.scatter(classA[0], classA[1])
    plt.scatter(classB[0], classB[1])

    # border lines
    #x = null_space(W_delta_batch[:, :-1])
    x = null_space(W_delta_batch[:, :])   # used for calculation without bias term
    xh = np.linspace(-1, 1)
    yh = xh * x[1]/x[0]
    #plt.plot(xh, yh+(W_delta_batch[0, 2]/np.linalg.norm(W_delta_batch)), label="delta_batch")
    plt.plot(xh, yh + (W_delta_batch[0, 1] / np.linalg.norm(W_delta_batch)), label="delta_batch")  # used for calculation without bias term

    #x = null_space(W_delta_nobatch[:, :-1])
    #xh = np.linspace(-1, 1)
    #yh = xh * x[1]/x[0]
    #plt.plot(xh, yh+(W_delta_nobatch[0, 2]/np.linalg.norm(W_delta_nobatch)), label="delta_nobatch")

    #x = null_space(W_perceptron[:, :-1])
    #xh = np.linspace(-1, 1)

    #yh = xh * x[1]/x[0]
    #plt.plot(xh, yh+(W_perceptron[0, 2]/np.linalg.norm(W_perceptron)), label="perceptron_batch")

    plt.legend()
    #plt.title('borderlines_learningrate{}.svg'.format(lr))
    plt.title('Using Delta Rule without bias term')
    plt.ylim(-4,4)
    plt.xlim(-4,4)
    plt.savefig('plots/3.1.2_no_bias/borderlines_learningrate{}.png'.format(lr))
    plt.clf()


loss_perceptron_mean = np.mean(loss_perceptron_all, axis=0)
loss_delta_batch_mean = np.mean(loss_delta_batch_all, axis=0)
loss_delta_nobatch_mean = np.mean(loss_delta_nobatch_all, axis=0)

plt.plot(np.arange(len(loss_delta_batch_mean)), loss_delta_batch_mean, label="delta_batch")
plt.plot(np.arange(len(loss_delta_nobatch_mean)), loss_delta_nobatch_mean, label="delta_nobatch")
plt.plot(np.arange(len(loss_perceptron_mean)), loss_perceptron_mean, label="perceptron_batch")
plt.legend()
plt.title('loss_LearningRate_{}'.format(lr))
#plt.savefig('plots/3.1.3/loss_learningrate_{}.svg'.format(lr))

print('Loop finished')