#########################                 DESCRIPTION                 ##############################

#Giving an input of 1xN where each point is generated using an laplacian distribution with parameter mean_input and b_input.
#Estimating the value of mean and b using these N points and generate a laplacian distribution using them.


import numpy as np
import random
import matplotlib.pyplot as plt

def merge(left_half,right_half):
	
	sort_list = []
	left_half_index = right_half_index = 0
	left_half_len, right_half_len = len(left_half), len(right_half)

	for i in range(left_half_len+right_half_len):
		if left_half_index < left_half_len and right_half_index < right_half_len:
			if left_half[left_half_index] <= right_half[right_half_index]:
				sort_list.append(left_half[left_half_index])
				left_half_index += 1
			else:
				sort_list.append(right_half[right_half_index])
				right_half_index += 1

		elif left_half_index == left_half_len:
			sort_list.append(right_half[right_half_index])
			right_half_index += 1

		elif right_half_index == right_half_len:
			sort_list.append(left_half[left_half_index])
			left_half_index += 1

	return sort_list

def mergesort(X):
	
	if len(X) <= 1:
		return X

	mid = len(X)//2

	left_half = mergesort(X[:mid])
	right_half = mergesort(X[mid:])

	return merge(left_half,right_half)


def median(X):
	sort_list = mergesort(X)
	if len(sort_list)%2 == 0:
		return (sort_list[len(sort_list)//2]+sort_list[(len(sort_list)//2)-1])/2
	else:
		return sort_list[len(sort_list)//2]



mean_input = random.uniform(0,10)
b_input = random.uniform(0,10)
N = N = np.random.randint(1000,1e4)

X = np.random.laplace(mean_input,b_input,N)

mean = median(list(X))
b = sum(np.array([np.absolute(x-mean) for x in X]))/N

Y = np.random.laplace(mean,b,N)
print("Input value of mean: " + str(mean_input))
print("Estimated value of mean: " + str(mean))
print("Input value of b: " + str(b_input))
print("Estimated value of b: " + str(b))

plt.figure(figsize=(100,100))
plt.subplot(1,2,1)
plt.hist(X,bins=int(N/4))
plt.subplot(1,2,2)
plt.hist(Y,bins=int(N/4))
plt.show()