#########################                 DESCRIPTION                 ##############################

#Giving an input of 1xN where each point is generated using an gaussian distribution with parameter mean_input and std_input.
#Estimating the value of mean and std using these N points and generate a gaussian distribution using them.


import numpy as np
import random
import matplotlib.pyplot as plt

mean_input = random.uniform(0,10)
std_input = random.uniform(0,10)
N = np.random.randint(1000,1e4)

X = np.random.normal(mean_input,std_input,N)

mean = np.mean(X)
std = np.sqrt(np.sum(np.square(X-mean))/N)

Y = np.random.normal(mean,std,N)
print("Input value of mean: " + str(mean_input))
print("Estimated value of mean: " + str(mean))
print("Input value of standard deviation: " + str(std_input))
print("Estimated value of standard deviation: " + str(std))

plt.figure(figsize=(100,100))
plt.subplot(1,2,1)
plt.hist(X,bins=int(N/4))
plt.subplot(1,2,2)
plt.hist(Y,bins=int(N/4))
plt.show()