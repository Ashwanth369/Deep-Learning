#########################                 DESCRIPTION                 ##############################

#Giving an input of 1xN where each point is generated using an poisson distribution with parameter lamda_input.
#Estimating the value of lamda using these N points and generate a poisson distribution using it.



import numpy as np
import random
import matplotlib.pyplot as plt

lamda_input = random.uniform(0,10)
N = np.random.randint(1e3,1e4)

X = np.random.poisson(lamda_input,N)

lamda = np.mean(X)
Y = np.random.poisson(lamda,N)

print("Input value of lamda: " + str(lamda_input))
print("Estimated value of lamda: " + str(lamda))


plt.figure(figsize=(100,100))
plt.subplot(1,2,1)
plt.hist(X,bins=20)
plt.subplot(1,2,2)
plt.hist(Y,bins=20)
plt.show()