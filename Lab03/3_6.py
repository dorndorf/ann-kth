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
    for i in range(9):
        train_data=data[0:i+1]
        hopf1.sparse_calc_weight_matrix(train_data, activity)
        x_before=train_data
        for j in range(max_iter):
            x_after=hopf1.sparse_synchr_update(x_before,bias)
            if np.all(x_after==x_before):
                print('iterations=',j+1)
                break
            else:
                x_before=x_after
        recall=x_after
        if np.array_equal(recall,train_data):
            max_trained_patterns = i+1
            print(max_trained_patterns)
        else:
            break
    return max_trained_patterns
            

data = loadData()
#Here we will use binary (0,1) patterns
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (data[i][j]==-1):
            data[i][j]=0

p1=data[0]
p2=data[1]
p3=data[2]
p = np.concatenate((p1, p2, p3)).reshape(-1, len(p1))
max_iter=20

hopf1 = networks.HopfieldNetwork(p)

#I don't know if I understand correctly. Please check.
#activity=0.1
activity=0.1
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
    
