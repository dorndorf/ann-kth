import numpy as np
import networks
import itertools

x1 = np.array([-1, -1,  1, -1,  1, -1, -1,  1])
x2 = np.array([-1, -1, -1, -1, -1,  1, -1, -1])
x3 = np.array([-1,  1,  1, -1, -1,  1, -1,  1])

x = np.concatenate((x1, x2, x3)).reshape(-1, len(x1))

hopf1 = networks.HopfieldNetwork(x)
hopf1.calc_weight_matrix(x)


x1d = np.array([1, -1,  1, -1,  1, -1, -1,  1])
x2d = np.array([1, 1, -1, -1, -1,  1, -1, -1])
x3d = np.array([1,  1,  1, -1, 1,  1, -1,  1])

xd = np.concatenate((x1d, x2d, x3d)).reshape(-1, len(x1))

max_iter=20

#check convergence
x_before=xd
for i in range(max_iter):
    x_after = hopf1.synchr_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

print(x_after)

for i in range(x_after.shape[0]):
    if np.all(x_after[i]==x[i]):
        print( 'new pattern and x',i+1, 'are same')
    else:
        print( 'new pattern and x',i+1, 'are not same')


#attractors. I do not know how to find attractors. 
all_patterns = [list(i) for i in itertools.product([-1, 1], repeat=8)]

#make the starting pattern even more dissimilar to the stored ones
x1m = np.array([1, 1,  -1, 1,  -1, -1, -1,  1])
x2m = np.array([1, 1, 1, 1, 1,  1, -1, -1])
x3m = np.array([1,  -1,  -1, 1, 1,  1, -1,  1])

xm = np.concatenate((x1m, x2m, x3m)).reshape(-1, len(x1))

x_before=xm
for i in range(max_iter):
    x_after = hopf1.synchr_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

print(x_after)

for i in range(x_after.shape[0]):
    if np.all(x_after[i]==x[i]):
        print( 'new pattern and x',i+1, 'are same')
    else:
        print( 'new pattern and x',i+1, 'are not same')



              








    
    
