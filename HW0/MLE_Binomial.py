#########################                 DESCRIPTION                 ##############################

#Giving an input of 1xN where each point is generated using an binomial distribution with parameter p.
#Estimating the value of prob using these N points and generate a binomial distribution using it.


import numpy as np
import random
import matplotlib.pyplot as plt

n = np.random.randint(10,1000)
p = random.uniform(0,1)
N = np.random.randint(1000,1e4)

X = np.random.binomial(n,p,N)

prob = np.mean(X)/n
print(prob,p)
Y = np.random.binomial(n,prob,N)

plt.figure(figsize=(100,100))
plt.subplot(1,2,1)
plt.hist(X,bins=int(n/4))
plt.xlim(0,n)
plt.subplot(1,2,2)
plt.hist(Y,bins=int(n/4))
plt.xlim(0,n)
plt.show()

