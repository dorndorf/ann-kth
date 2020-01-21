import numpy as np

patterns = np.array([[-1, 1, -1, 1],
                    [-1, -1, 1, 1],
                    [1, 1, 1, 1]])

targets = np.array([-1, 1, 1, -1])[np.newaxis, :]

W = np.random.normal(0.0, 1.0, size=(targets.shape[0], patterns.shape[0]))

lr = 0.001
epochs = 20

for ep in range(epochs):
    delta_W = -lr*np.matmul(np.matmul(W, patterns) - targets, patterns.transpose())
    W += delta_W
    print("Epoch {}: New W is {}".format(ep, W))

result = np.matmul(W, patterns)

print("\nFinals result for W*Input = {}".format(result))
print("Real result should have been [-1, 1, 1, -1]")