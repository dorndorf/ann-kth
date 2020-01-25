import numpy as np

patterns = np.array([[-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [1, 1, 1, 1]])

targets = np.array([-1, 1, 1, -1])[np.newaxis, :]

#print(patterns.shape)
#print(targets.shape)

W = np.random.normal(0.0, 0.1, size=(targets.shape[0], patterns.shape[0]))

lr = 0.001
epochs = 10

# for ep in range(epochs):
#     delta_W = -lr*np.matmul(np.matmul(W, patterns) - targets, patterns.transpose())
#     W += delta_W
#     print("Epoch {}: New W is {}".format(ep, W))

# result = np.matmul(W, patterns)
#
# print("\nFinal result for W*Input = {}".format(result))
# print("Real result should have been [-1, -1, -1, 1]")


def phi_function(x):
    return (2 / 1 + np.exp(-x)) - 1

def phi_dev(x):
    return (1+phi_function(x))*(1-phi_function(x))/2

num_hidden = 4
W = np.random.normal(0.0, 1, size=(num_hidden, patterns.shape[0]))
V = np.random.normal(0.0, 1, size=(targets.shape[0], num_hidden+1))
ndata = patterns.shape[1]
alpha = 0.9
theta = 1.0
psi = 1.0
epochs = 1
lr = 0.01

for ep in range(epochs):
    ## Forward Pass
    hin = np.matmul(W, patterns)
    #print("Shape hin: {}".format(hin))
    hout = np.concatenate((phi_function(hin), np.ones(ndata)[np.newaxis, :]))
    #print("Shape hout: {}".format(hout))
    oin = np.matmul(V, hout)
    #print("Shape oin: {}".format(oin))
    out = phi_function(oin)
    print("Out: {}".format(out))


    ##Backward Pass
    delta_o = (out - targets) * phi_dev(out)
    print("Delta_o:{}".format(delta_o.shape))
    delta_h = (np.matmul(np.transpose(V), delta_o) * phi_dev(hout))[:num_hidden, :]
    print("Delta_h:{}".format(delta_h.shape))

    ## New Learning Rate
    theta = alpha*theta - (1 - alpha) * np.matmul(delta_h, np.transpose(patterns))
    psi = alpha*psi - (1 - alpha) * np.matmul(delta_o, np.transpose(hout))

    delta_W = - lr * theta * np.matmul(delta_h, np.transpose(patterns))
    print("Delta_W:{}".format(delta_W.shape))
    delta_V = - lr * psi * np.matmul(delta_o, np.transpose(hout))
    print("Delta_V:{}".format(delta_V.shape))

    W += delta_W
    V += delta_V