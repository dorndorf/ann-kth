import numpy as np
import matplotlib.pyplot as plt
import learning_functions
import learning_rules
import time

x_train = np.arange(0, 2*np.pi, 0.1)[:, np.newaxis]
y_train_sin = np.sin(2*x_train)
y_train_square = np.where(y_train_sin >= 0.0, 1, -1).astype(float)

y_train_sin += np.random.normal(0.0, 0.1, y_train_sin.shape[0])[:, np.newaxis]
y_train_square += np.random.normal(0.0, 0.1, y_train_square.shape[0])[:, np.newaxis]

x_test = np.arange(0.05, 2*np.pi, 0.1)[:, np.newaxis]
y_test_sin = np.sin(2*x_test)
y_test_square = np.where(y_test_sin >= 0.0, 1, -1).astype(float)

y_test_sin += np.random.normal(0.0, 0.1, y_test_sin.shape[0])[:, np.newaxis]
y_test_square += np.random.normal(0.0, 0.1, y_test_square.shape[0])[:, np.newaxis]



# RBF network: batch learning
means = np.arange(0, 2*np.pi, 0.25)[:, np.newaxis]
vars = np.ones(means.shape[0]) * 0.2
weights = np.zeros(means.shape[0])[:, np.newaxis]

first_rbf = learning_functions.GaussianRBF(means, vars, weights)

tic = time.perf_counter()
first_rbf.least_squares(x_train, y_train_square)
toc = time.perf_counter()
timepass_rbf = toc - tic
print("Calculation time RBF-batch: {}".format(timepass_rbf))
print("Squared Residual Error RBF-batch: {}".format(first_rbf.res_error(x_test, y_test_square)))

output_rbf_batch = first_rbf.forward_pass(x_test)

# MLP network: batch learning
patterns = np.append(np.transpose(x_train),np.ones(len(x_train))[np.newaxis,:],axis = 0)
targets = np.transpose(y_train_square)
lr = 0.04
epochs = 10000
alpha = 0.8
num_hidden = 26
batch = True

tic = time.perf_counter()
W, V,  loss_nobatch = learning_rules.two_layer_backprop(patterns, targets, epochs, lr, alpha, num_hidden, loss_name="rmse")
toc = time.perf_counter()
timepass_mlp = toc - tic
print("Calculation time MLP-batch: {}".format(timepass_mlp))

patterns = np.append(np.transpose(x_test),np.ones(len(x_test))[np.newaxis,:],axis = 0)
ndata = patterns.shape[1]
hin = np.matmul(W, patterns)
hout = np.concatenate((learning_rules.phi_function(hin), np.ones(ndata)[np.newaxis, :]))
oin = np.matmul(V, hout)
output_mlp_batch = np.transpose(learning_rules.phi_function(oin))

SquaredResidualError_MLP = np.divide(np.linalg.norm(output_mlp_batch - y_test_square, 2),y_test_square.shape[0])
print("Squared Residual Error MLP-batch: {}".format(SquaredResidualError_MLP))

# plot
label_rbf = 'RBF: {0:.3f} sec'.format(timepass_rbf)
label_mlp = 'MLP: {0:.3f} sec'.format(timepass_mlp)
plt.plot(x_test, y_test_sin, label = 'testdata')
plt.plot(x_test, output_rbf_batch, label = label_rbf)
plt.plot(x_test, output_mlp_batch, label = label_mlp)
plt.legend()
plt.title('Square approximated with MLP and RBF')
plt.show()
#plt.savefig('plots/ErrorSinSomWithoutNoise.png')

print('finished')


