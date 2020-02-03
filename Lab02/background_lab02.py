import numpy as np
import matplotlib.pyplot as plt

x_train = np.arange(0, 2*np.pi, 0.1)
y_train_sin = np.sin(2*x_train)
y_train_square = np.where(y_train_sin >= 0.0, 1, -1)

x_test = np.arange(0.05, 2*np.pi, 0.1)
y_test_sin = np.sin(2*x_test)
y_test_square = np.where(y_test_sin >= 0.0, 1, -1)



