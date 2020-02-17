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
p4 = data[3]
p5 = data[4]
p6 = data[5]
p7 = data[6]
p10=data[9]
p11=data[10]

data = np.random.randint(0, 2, size=(300, 300))
data[data==0.0] = -1

num_att = 150

r1, r2, r3, r4 = [], [], [], []

for r in range(4):
    result = []
    for num_att in range(1, 299, 20):
        random_indices = random.sample(range(299), num_att)

        p = np.concatenate(data[random_indices]).reshape(-1, 300)

        hopf1 = networks.HopfieldNetwork(p)
        hopf1.calc_weight_matrix(p)
        max_iter=20


        #Check that the three patterns are stable.
        x_before=p
        for i in range(max_iter):
            x_after = hopf1.asynchr_update(x_before)
            if np.all(x_after == x_before):
                print('iterations=',i+1)
                break
            else:
                x_before=x_after

        stable_counter = 0
        for i in range(x_after.shape[0]):
            if np.all(x_after[i]==p[i]):
                print( 'p',i+1, 'is stable')
                stable_counter += 1
            else:
                print( 'p',i+1, 'is not stable')
            #display_pattern(x_after[i])

        result.append(stable_counter)

    result_noise = []
    for num_att in range(1, 299, 20):
        random_indices = random.sample(range(299), num_att)

        p = np.concatenate(data[random_indices]).reshape(-1, 300)

        hopf1 = networks.HopfieldNetwork(p)
        hopf1.calc_weight_matrix(p)
        max_iter=20


        #Check that the three patterns are stable.
        x_before=add_noise(p, 10)
        for i in range(max_iter):
            x_after = hopf1.asynchr_update(x_before)
            if np.all(x_after == x_before):
                print('iterations=',i+1)
                break
            else:
                x_before=x_after

        stable_counter = 0
        for i in range(x_after.shape[0]):
            if np.all(x_after[i]==p[i]):
                print( 'p',i+1, 'is stable')
                stable_counter += 1
            else:
                print( 'p',i+1, 'is not stable')
            #display_pattern(x_after[i])

        result_noise.append(stable_counter)

    result_wii = []
    for num_att in range(1, 299, 20):
        random_indices = random.sample(range(299), num_att)

        p = np.concatenate(data[random_indices]).reshape(-1, 300)

        hopf1 = networks.HopfieldNetwork(p)
        hopf1.calc_weight_matrix(p)
        max_iter=20
        hopf1.remove_weight_matrix_diag()

        #Check that the three patterns are stable.
        x_before=p
        for i in range(max_iter):
            x_after = hopf1.asynchr_update(x_before)
            if np.all(x_after == x_before):
                print('iterations=',i+1)
                break
            else:
                x_before=x_after

        stable_counter = 0
        for i in range(x_after.shape[0]):
            if np.all(x_after[i]==p[i]):
                print( 'p',i+1, 'is stable')
                stable_counter += 1
            else:
                print( 'p',i+1, 'is not stable')
            #display_pattern(x_after[i])

        result_wii.append(stable_counter)

    result_noise_wii = []
    for num_att in range(1, 299, 20):
        random_indices = random.sample(range(299), num_att)

        p = np.concatenate(data[random_indices]).reshape(-1, 300)

        hopf1 = networks.HopfieldNetwork(p)
        hopf1.calc_weight_matrix(p)
        max_iter=20

        hopf1.remove_weight_matrix_diag()


        #Check that the three patterns are stable.
        x_before=add_noise(p, 10)
        for i in range(max_iter):
            x_after = hopf1.asynchr_update(x_before)
            if np.all(x_after == x_before):
                print('iterations=',i+1)
                break
            else:
                x_before=x_after

        stable_counter = 0
        for i in range(x_after.shape[0]):
            if np.all(x_after[i]==p[i]):
                print( 'p',i+1, 'is stable')
                stable_counter += 1
            else:
                print( 'p',i+1, 'is not stable')
            #display_pattern(x_after[i])

        result_noise_wii.append(stable_counter)

    r1.append(result)
    r2.append(result_noise)
    r3.append(result_wii)
    r4.append(result_noise_wii)


plt.plot(range(1, 299, 20), np.mean(np.array(r1), axis=0), label='No Noise')
plt.fill_between(range(1, 299, 20), np.mean(np.array(r1), axis=0)-np.var(np.array(r1), axis=0),
                 np.mean(np.array(r1), axis=0)+np.var(np.array(r1), axis=0), alpha=0.5)
plt.plot(range(1, 299, 20), np.mean(np.array(r2), axis=0), label='Noise')
plt.fill_between(range(1, 299, 20), np.mean(np.array(r2), axis=0)-np.var(np.array(r2), axis=0),
                 np.mean(np.array(r2), axis=0)+np.var(np.array(r2), axis=0), alpha=0.5)
plt.plot(range(1, 299, 20), np.mean(np.array(r3), axis=0), label='No Noise, No Diagonal')
plt.fill_between(range(1, 299, 20), np.mean(np.array(r3), axis=0)-np.var(np.array(r3), axis=0),
                 np.mean(np.array(r3), axis=0)+np.var(np.array(r3), axis=0), alpha=0.5)
plt.plot(range(1, 299, 20), np.mean(np.array(r4), axis=0), label='Noise, No Diagonal')
plt.fill_between(range(1, 299, 20), np.mean(np.array(r4), axis=0)-np.var(np.array(r4), axis=0),
                 np.mean(np.array(r4), axis=0)+np.var(np.array(r4), axis=0), alpha=0.5)
plt.xlabel('number of learned patterns')
plt.ylabel('number of stable patterns')
plt.legend()
plt.show()

#print('Mean: {}'.format(np.mean(np.array(result)/num_att)))
#print('Variance: {}'.format(np.var(np.array(result)/num_att)))