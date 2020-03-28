import matplotlib.pyplot as plt
import numpy as np
import random
import networks

def loadData():
    with open("data/pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

def display_pattern(pattern):
  pattern=pattern.reshape(32,32)
  plt.imshow(pattern)
  plt.show()

def add_noise(pattern, noise_level):
  d= len(pattern)
  res=np.copy(pattern)
  random_indexes = random.sample(list(np.arange(d)), int(noise_level *d / 100))
  for index in random_indexes:
    res[index] = - pattern[index]
  return(res)


data = loadData()
p1=data[0]
p2=data[1]
p3=data[2]
p10=data[9]
p11=data[10]
p = np.concatenate((p1, p2, p3)).reshape(-1, len(p1))

hopf1 = networks.HopfieldNetwork(p)
hopf1.calc_weight_matrix(p)
max_iter=20

'''
#add noise to p1
for i in [0,10,20,30,40,50,60,70,80,90,100]:
  p1_noise = add_noise(p1,i)
  x_before=p1_noise
  for j in range(max_iter):
    x_after = hopf1.synchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',j+1)
        break
    else:
        x_before=x_after
  
  if (np.array_equal(p1,x_after)):
    print(i,'%')
    print("They're the same")
    display_pattern(p1_noise)
    display_pattern(x_after)
  else :
    print("They're different")
    display_pattern(p1_noise)
    display_pattern(x_after)

#add noise to p2
for i in [0,10,20,30,40,50,60,70,80,90,100]:
  p2_noise = add_noise(p2,i)
  x_before=p2_noise
  for j in range(max_iter):
    x_after = hopf1.synchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',j+1)
        break
    else:
        x_before=x_after
  
  if (np.array_equal(p2,x_after)):
    print(i,'%')
    print("They're the same")
    display_pattern(p2_noise)
    display_pattern(x_after)
  else :
    print("They're different")
    display_pattern(p2_noise)
    display_pattern(x_after)

'''
#add noise to p3
for i in [0,10,20,30,40,50,60,70,80,90,100]:
  p3_noise = add_noise(p3,i)
  x_before=p3_noise
  for j in range(max_iter):
    x_after = hopf1.synchr_one_update(x_before)
    if np.all(x_after==x_before):
        print('iterations=',j+1)
        break
    else:
        x_before=x_after
  
  if (np.array_equal(p3,x_after)):
    print(i,'%')
    print("They're the same")
    display_pattern(p3_noise)
    display_pattern(x_after)
  else :
    print("They're different")
    display_pattern(p3_noise)
    display_pattern(x_after)

#results 30%/40% could be removed. attractors: p1-3, inverse of p1-3, another one and inverse of it 





