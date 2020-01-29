import numpy as np
import plot_data

def delta_rule(patterns, targets, epochs, lr, batch=True):
    W = np.random.normal(0.0, 0.1, size=(targets.shape[0], patterns.shape[0]))
    loss = []
    if batch:
        for ep in range(epochs):
            false_class = np.count_nonzero(np.sign(np.matmul(W, patterns)) - targets != 0.)
            loss.append(false_class/targets.shape[1])
            #plot_data.plot_borderlines(patterns, targets, W, loss)
            delta_W = -lr * np.matmul(np.matmul(W, patterns) - targets, patterns.transpose())
            W += delta_W
            #print("Epoch {}: New W is {}".format(ep, W))

    else:
        for ep in range(epochs):
            false_class = np.count_nonzero(np.sign(np.matmul(W, patterns)) - targets != 0.)
            loss.append(false_class / targets.shape[1])
            for sample in range(patterns.shape[1]):
                #print(patterns[:, sample].shape)
                delta_W = -lr * np.dot(np.dot(W[0], patterns[:, sample]) - targets[0, sample], patterns[:, sample])
                W += delta_W
            #print("Epoch {}: New W is {}".format(ep, W))
    return W, loss

def perceptron_rule(patterns, targets, epochs, lr, batch=True):
    W = np.random.normal(0.0, 0.1, size=(targets.shape[0], patterns.shape[0]))
    loss = []
    for ep in range(epochs):
        pred = np.matmul(W, patterns)
        pred[pred > 0] = 1.
        pred[pred <= 0] = 0.
        false_class = np.count_nonzero(pred - targets != 0.)
        loss.append(false_class / targets.shape[1])
        delta_W = lr * np.matmul(targets - pred, patterns.transpose())
        W += delta_W
        #print("Epoch {}: New W is {}".format(ep, W))
    return W, loss

def two_layer_backprop(patterns, targets, epochs, lr, alpha=0.9, num_hidden=64, loss_name="false_class", batch=True):
    theta = 1.0
    psi = 1.0
    loss = []
    W = np.random.randn(num_hidden, patterns.shape[0]) * np.sqrt(2/(patterns.shape[0]))
    V = np.random.randn(targets.shape[0], num_hidden + 1) * np.sqrt(2/(num_hidden + 1))
    for ep in range(epochs):
        ndata = patterns.shape[1]
        ## Forward Pass
        hin = np.matmul(W, patterns)
        # print("Shape hin: {}".format(hin))
        hout = np.concatenate((phi_function(hin), np.ones(ndata)[np.newaxis, :]))
        # print("Shape hout: {}".format(hout))
        oin = np.matmul(V, hout)
        # print("Shape oin: {}".format(oin))
        out = phi_function(oin)
        #print("Out: {}".format(out))

        if loss_name == "rmse":
            loss.append(np.sqrt(np.mean((out - targets) ** 2)))
        else:
            false_class = np.count_nonzero(np.sign(out) - targets != 0.)
            loss.append(false_class / targets.shape[1])

        ##Backward Pass
        delta_o = (out - targets) * phi_dev(out)
        #print("Delta_o:{}".format(delta_o.shape))
        delta_h = (np.matmul(np.transpose(V), delta_o) * phi_dev(hout))[:-1, :]
        #print("Delta_h:{}".format(delta_h.shape))

        ## New Learning Rate
        theta = alpha * theta - (1 - alpha) * np.matmul(delta_h, np.transpose(patterns))
        psi = alpha * psi - (1 - alpha) * np.matmul(delta_o, np.transpose(hout))

        delta_W = lr * theta
        #print("Delta_W:{}".format(delta_W))
        delta_V = lr * psi
        # print("Delta_V:{}".format(delta_V))

        W += delta_W
        V += delta_V
    return W, V, loss

def phi_function(x):
    return (2 / (1 + np.exp(-x))) - 1

def phi_dev(x):
    return (1+phi_function(x))*(1-phi_function(x)) * 0.5

def relu(x):
    x[x < 0.0] = 0.0
    return x

def relu_dev(x):
    x[x < 0.0] = 0.0
    x[x > 0.0] = 1.0
    return x
