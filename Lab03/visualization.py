import numpy as np
import matplotlib.pyplot as plt

def plot_true_and_update(patterns, hopf, num_update=1, stepwidth=1, asynch=False):
    out = np.copy(patterns)
    for u in range(1, num_update+1):
        if asynch:
            out = hopf.asynchr_update(out)
        else:
            out = hopf.synchr_update(out)

        if u == 1 or u % stepwidth == 0:
            num_patterns = len(out)
            fig, axes = plt.subplots(2, num_patterns)
            if num_patterns > 1:
                for i in range(num_patterns):
                    axes[0, i].imshow(patterns[i].reshape(32, 32), cmap='binary')
                for i in range(num_patterns):
                    axes[1, i].imshow(out[i].reshape(32, 32), cmap='binary')
            else:
                axes[0].imshow(patterns.reshape(32, 32), cmap='binary')
                axes[1].imshow(out.reshape(32, 32), cmap='binary')
            fig.suptitle('Update {}'.format(u))
            plt.show()

def plot_async_update(patterns, hopf, stepwidth=100):
    out = np.copy(patterns)
    counter = 0

    for p in range(patterns.shape[0]):
        for i in range(patterns.shape[1]):
            out[p, i] = np.sign(np.sum(hopf.W[:, i] * out[p]))
            counter += 1
            if counter % stepwidth == 0:
                fig, axes = plt.subplots(2, 1)
                axes[0].imshow(patterns.reshape(32, 32), cmap='binary')
                axes[1].imshow(out.reshape(32, 32), cmap='binary')
                fig.suptitle('Iteration {}'.format(counter))
                plt.show()