import matplotlib.pyplot as plt
import numpy as np
import networks

def loadData():
    with open("pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

def display_pattern(pattern):
  pattern=pattern.reshape(32,32)
  plt.imshow(pattern)
  plt.show()

def max_trained_patterns(bias, activity):
    max_trained_patterns = 0
    train_data=data
    hopf1.sparse_calc_weight_matrix(train_data)
    x_before=train_data
    for j in range(max_iter):
        x_after=hopf1.sparse_synchr_update(x_before, bias)
        if np.all(x_after==x_before):
            print('iterations=',j+1)
            break
        else:
            x_before=x_after
    recall=x_after

    for row, val in enumerate(recall):
        if np.array_equal(val,train_data[row]):
            max_trained_patterns += 1
            #print(max_trained_patterns)

    return max_trained_patterns
            

max_iter=20

data = np.zeros((300, 100))

num_sp = int(300 * 100 * 0.01)

data[np.random.randint(0, 300, num_sp), np.random.randint(0, 100, num_sp)] = 1.0

p = np.copy(data)

hopf1 = networks.HopfieldNetwork(p)

#I don't know if I understand correctly. Please check.
#activity=0.1
activity=0.1
biases = np.arange(0, 1.00, 0.05)
trained = []
for bias in biases:
    n=max_trained_patterns(bias,activity)
    trained.append(n)

plt.plot(biases, trained)
plt.xlabel("Bias")
plt.ylabel("Number of stored images")
plt.show()

'''
#activity=0.05
activity=0.05
biases = np.arange(0,2.01,0.05)
trained = []
for bias in biases:
    n=max_trained_patterns(bias,activity)
    trained.append(n)

plt.plot(biases, trained)
plt.xlabel("Bias")
plt.ylabel("Number of stored images")
plt.show()


#activity=0.01
activity=0.01
biases = np.arange(0,2.01,0.05)
trained = []
for bias in biases:
    n=max_trained_patterns(bias,activity)
    trained.append(n)

plt.plot(biases, trained)
plt.xlabel("Bias")
plt.ylabel("Number of stored images")
plt.show()
'''
    
