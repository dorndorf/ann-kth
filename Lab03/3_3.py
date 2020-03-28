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

hopf1 = networks.HopfieldNetwork(p)
hopf1.calc_weight_matrix(p)

#the energy at the different attractors
print('the energy at p1 is',hopf1.calc_energy(p1))
print('the energy at p2 is',hopf1.calc_energy(p2))
print('the energy at p3 is',hopf1.calc_energy(p3))

#the energy at the points of the distorted patterns
print('the energy at p10 is',hopf1.calc_energy(p10))
print('the energy at p11 is',hopf1.calc_energy(p11))

#sequential p11.
max_iter=20
x_before=p11
energy=[]
energy.append(hopf1.calc_energy(p11))
for i in range(max_iter):
    x_after = hopf1.asynchr_one_update(x_before)
    energy.append(hopf1.calc_energy(x_after))
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

print(energy)

plt.plot(range(i+2),energy)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("p11 (sequential update)")
plt.show()

#random weight matrix
N1=1024
random_W = np.random.normal(-1, 1, (N1, N1))
#np.fill_diagonal(random_W, 0)


hopf2 = networks.HopfieldNetwork(p)
hopf2.W = random_W

max_iter=20
x_before=p11
energy=[]
energy.append(hopf2.calc_energy(p11))
for i in range(max_iter):
    x_after = hopf2.asynchr_one_update(x_before)
    energy.append(hopf2.calc_energy(x_after))
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

print(energy)

plt.plot(range(i+2),energy)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("p11 (random weight matrix)")
plt.show()

#symmetric weight matrix
W_symmetric=0.5*(random_W+random_W.T)
hopf3 = networks.HopfieldNetwork(p)
hopf3.W = W_symmetric

max_iter=20
x_before=p11
energy=[]
energy.append(hopf3.calc_energy(p11))
for i in range(max_iter):
    x_after = hopf3.asynchr_one_update(x_before)
    energy.append(hopf3.calc_energy(x_after))
    if np.all(x_after==x_before):
        print('iterations=',i+1)
        break
    else:
        x_before=x_after

print(energy)

plt.plot(range(i+2),energy)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("p11 (symmetric weight matrix)")
plt.show()






