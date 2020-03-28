import matplotlib.pyplot as plt
import numpy as np
import networks

def loadData():
    with open("data/pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

def display_pattern(pattern):
  pattern=pattern.reshape(32,32)
  plt.imshow(pattern)
  plt.show()

data = loadData()
p1=data[0]
p2=data[1]
p3=data[2]
p10=data[9]
p11=data[10]
p = np.concatenate((p1, p2, p3)).reshape(-1, len(p1))

display_pattern(p10)

hopf1 = networks.HopfieldNetwork(p)
hopf1.calc_weight_matrix(p)

max_iter=20

#Check that the three patterns are stable.
x_before=p
for i in range(max_iter):
    x_after = hopf1.synchr_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

for i in range(x_after.shape[0]):
    if np.all(x_after[i]==p[i]):
        print( 'p',i+1, 'is stable')
    else:
        print( 'p',i+1, 'is not stable')

#Try the pattern p10, which is a degraded version of p1
x_before=p10
for i in range(max_iter):
    x_after = hopf1.synchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p1):
        print( 'p10 and p1 are same')
else:
        print( 'p10 and p1 are not same')

plt.title('new p10')
display_pattern(x_after)

#Try the pattern p11, which is a mixture of p2 and p3
x_before=p11
for i in range(max_iter):
    x_after = hopf1.synchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p2):
        print( 'p11 and p2 are same')
else:
        print( 'p11 and p2 are not same')

if np.all(x_after==p3):
        print( 'p11 and p3 are same')
else:
        print( 'p11 and p3 are not same')

plt.title('new p11')
display_pattern(x_after)


#random p10. I am not sure about random_one_update 
x_before=p10
for i in range(max_iter):
    x_after = hopf1.random_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p1):
        print( 'p10 and p1 are same')
else:
        print( 'p10 and p1 are not same')

plt.title('new p10')
display_pattern(x_after)

#random p11. I am not sure about random_one_update 
x_before=p11
for i in range(max_iter):
    x_after = hopf1.random_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p2):
        print( 'p11 and p2 are same')
else:
        print( 'p11 and p2 are not same')

if np.all(x_after==p3):
        print( 'p11 and p3 are same')
else:
        print( 'p11 and p3 are not same')

plt.title('new p11')
display_pattern(x_after)

    
#sequential p10.
x_before=p10
for i in range(max_iter):
    x_after = hopf1.asynchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p1):
        print( 'p10 and p1 are same')
else:
        print( 'p10 and p1 are not same')

plt.title('new p10')
display_pattern(x_after)

#sequential p11.
x_before=p11
for i in range(max_iter):
    x_after = hopf1.asynchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

if np.all(x_after==p2):
        print( 'p11 and p2 are same')
else:
        print( 'p11 and p2 are not same')

if np.all(x_after==p3):
        print( 'p11 and p3 are same')
else:
        print( 'p11 and p3 are not same')

plt.title('new p11')
display_pattern(x_after)





    

