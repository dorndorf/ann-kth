import numpy as np
import matplotlib.pyplot as plt
import learning_functions
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

#mean_distance = np.array([0.5,0.45,0.4,0.35,0.3,0.25,0.22,0.2,0.18])
#mean_distance = np.array([0.5,0.45,0.4,0.35,0.3,0.25])
mean_distance = np.array([1,0.8,0.6,0.5,0.4,0.35,0.32,0.3,0.27,0.25,0.22,0.2,0.18,0.15])
vars_values = np.array([0.3,0.2,0.1,0.05])

#best configuration for RBF (compromise between both functions)
mean_distance = np.array([0.25])
vars_values = np.array([0.2])

lr = 0.02
epochs = 100
squared_res_error_batch_all = np.zeros((len(vars_values),len(mean_distance)))
squared_res_error_delta_all = np.zeros((len(vars_values),len(mean_distance)))

timepass = []

for a in range(len(vars_values)):
    squared_res_error_batch = []
    squared_res_error_delta = []
    nodes = []
    for b in range(len(mean_distance)):
        squared_res_error_batch_temp = []
        squared_res_error_delta_temp = []
        for c in range(10):
            means = np.arange(0, 2*np.pi, mean_distance[b])[:, np.newaxis]
            vars = np.ones(means.shape[0]) * vars_values [a]
            weights = np.zeros(means.shape[0])[:, np.newaxis]

            # random positioning
            means = np.random.uniform(x_train[0],x_train[-1],len(means)) [:, np.newaxis]

            # batch learning (time investigation added)
            tic = time.perf_counter()
            first_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)
            toc = time.perf_counter()
            timepass.append(toc - tic)

            first_rbf.least_squares(x_train, y_train_sin)

            output_batch = first_rbf.forward_pass(x_test)
            squared_res_error_batch_temp.append(first_rbf.res_error(x_test, y_test_sin))

            #plt.plot(x_test, y_test_sin)
            #plt.plot(x_test, output_batch)
            #plt.show()

            # delta learning
            #weights = np.zeros(means.shape[0])[:, np.newaxis]
            #second_rbf = learning_functions.GaussianRBF(means, vars, weights, lr)
            #for ep in range(epochs):
             #   second_rbf.delta_rule(x_train, y_train_sin)

            #output_delta = second_rbf.forward_pass(x_test)
            #squared_res_error_delta_temp.append(second_rbf.res_error(x_test, y_test_sin))

            #plt.plot(x_test, y_test_sin)
            #plt.plot(x_test, output_delta)
            #plt.show()

            #print("Batch: {0:.4f}".format(squared_res_error_batch_temp[-1]))
            #print("Delta: {0:.4f}".format(squared_res_error_delta_temp[-1]))
            #print("Batch: {0:.4f} ({1:.4f})".format(np.mean(np.array(squared_res_error_batch)), np.std(np.array(squared_res_error_batch))))
            #print("Delta: {0:.4f} ({1:.4f})".format(np.mean(np.array(squared_res_error_delta)), np.std(np.array(squared_res_error_delta))))
            #print("Number of nodes: {}".format(first_rbf.n))
            #print("Width of nodes: {}" .format(vars_values[a]))

        nodes.append(first_rbf.n)
        squared_res_error_batch.append(np.mean(np.array(squared_res_error_batch_temp)))
        #squared_res_error_delta.append(np.mean(np.array(squared_res_error_delta_temp)))

        #print("Batch: {0:.4f}".format(squared_res_error_batch_temp[-1]))
        #print("Delta: {0:.4f}".format(squared_res_error_delta_temp[-1]))
        print("Batch: {0:.4f} ({1:.4f})".format(np.mean(np.array(squared_res_error_batch_temp)), np.std(np.array(squared_res_error_batch_temp))))
        #print("Delta: {0:.4f} ({1:.4f})".format(np.mean(np.array(squared_res_error_delta_temp)), np.std(np.array(squared_res_error_delta_temp))))
        print("Number of nodes: {}".format(first_rbf.n))
        print("Width of nodes: {}".format(vars_values[a]))

    squared_res_error_batch_all[a,:] = np.array(squared_res_error_batch)
    #squared_res_error_delta_all[a,:] = np.array(squared_res_error_delta)

nodes = np.array(nodes)
for z in range (len(vars_values)):
    label1 = "batch, width: {}".format(vars_values[z])
    #label2 = "delta, width: {}".format(vars_values[z])
    plt.plot(nodes, squared_res_error_batch_all[z,:], label= label1)
    #plt.plot(nodes, squared_res_error_delta_all[z,:], label= label2)

plt.legend()
plt.title('Squared Residual error over nodes: sin with random positioning')
#plt.ylim((0.02,0.09))
#plt.show()
#plt.savefig('plots/ErrorSinRandomPositioningIncreasedWidth.png')

print('finished')
print("Execution-Time-RBF-Batch: {0:.4f} ({1:.4f})".format(np.mean(np.array(timepass)), np.std(np.array(timepass))))
print(timepass)



