import numpy as np
import networks

x1 = np.array([-1, -1,  1, -1,  1, -1, -1,  1])
x2 = np.array([-1, -1, -1, -1, -1,  1, -1, -1])
x3 = np.array([-1,  1,  1, -1, -1,  1, -1,  1])

x = np.concatenate((x1, x2, x3)).reshape(-1, len(x1))

hopf1 = networks.HopfieldNetwork(x)
hopf1.calc_weight_matrix(x)

x1 = np.array([-1, -1,  1, -1,  1, -1, -1,  1])
x2 = np.array([-1, -1, -1, -1, -1,  1, 1, -1])
x3 = np.array([-1,  1,  1, -1, -1,  1, -1,  1])

xd = np.concatenate((x1, x2, x3)).reshape(-1, len(x1))

#xd may be wrong. 
'''
x1d = np.array([1, -1,  1, -1,  1, -1, -1,  1])
x2d = np.array([1, 1, -1, -1, -1,  1, -1, -1])
x3d = np.array([1,  1,  1, -1, 1,  1, -1,  1])

xd = np.concatenate((x1d, x2d, x3d)).reshape(-1, len(x1))
'''
out = hopf1.asynchr_update(xd)

print(out)

