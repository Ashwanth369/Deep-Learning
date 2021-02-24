import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from HW0 import K_MEANS

def readfile(fname):
	data = []
	f1 = open(fname,"r")
	f2 = f1.readlines()
	
	for line in f2:
		row = list(map(float,line.split()))
		data.append(row)

	data = np.array(data)
	return data

def gmm(X,cov,mean,weight,d):

	det = abs(np.asscalar(np.linalg.det(cov)))
	i=0
	while det!=0 and i<cov.shape[0]: 
		cov[i][i] += 0.1
		i += 1
		det = abs(np.asscalar(np.linalg.det(cov)))
	cov_ = np.linalg.inv(cov)
	A = np.array(X-mean).reshape((d,-1))
	exponent = np.asscalar(np.dot(np.dot(A.T,cov_),A))
	res = np.exp(-0.5*exponent)/(np.sqrt(((2*np.pi)**d)*abs(det)))
	return res*weight

def loglikelihood(X_input,covs,means,weights,K,N,d):
	summ1 = 0
	for i in range(N):
		summ2 = 0
		for j in range(K):
			summ2 += gmm(X_input[:,i],covs[j],means[j],weights[j],d)

		summ1 += np.log(summ2)
	return summ1

def initialize(X_input,d,N,K):

	# Y_output = K_MEANS.KMeans(n_clusters=K).fit(X_input)
	# weights = np.empty(K)
	# mean_vectors = np.empty((K,d))
	# covariance_matrices = np.empty((K,d,d))
	
	# for i in range(K):
	# 	temp = []
	# 	count = 0
	# 	for j in range(N):
	# 		if Y_output.labels_[j] == i:
	# 			count += 1
	# 			temp.append(X_input[:,j])

	# 	temp = np.array(temp)
	# 	weights[i] = count/N
	# 	mean_vectors[i] = np.sum(temp,axis=0)/count
	# 	covariance_matrices[i] = 50*np.cov(temp.T)

	# # plt.scatter(X_input[0], X_input[1], c=Y_output.labels_, s=40, cmap='viridis')
	# # plt.show()

	weights = np.ones(K)/K
	mean_vectors = np.zeros((K,d))
	covariance_matrices = np.random.rand(K,d,d)*5000

	print(weights)
	print(mean_vectors)
	print(covariance_matrices)
	print("\n\n\n\n\n\n")
	return weights,mean_vectors,covariance_matrices


def GMM(X_input,K):
	
	d,N = X_input.shape[0],X_input.shape[1]
	weights,mean_vectors,covariance_matrices = initialize(X_input,d,N,K)

	loglikelihood_new = loglikelihood(X_input,covariance_matrices,mean_vectors,weights,K,N,d)
	loglikelihood_old = -100
	gamma = np.zeros((N,K))
	iteration =1

	while abs(loglikelihood_new - loglikelihood_old) > 0.001:
		print("---------------------Iteration "+str(iteration)+"---------------------")
		iteration += 1

		gammatot = np.zeros((N,1))
		
		for i in range(N):
			for j in range(K):
				gammatot[i] += gmm(X_input[:,i],covariance_matrices[j],mean_vectors[j],np.asscalar(weights[j]),d)
		
		print(gammatot[0])
		for i in range(N):
			for j in range(K):
				gamma[i][j] = gmm(X_input[:,i],covariance_matrices[j],mean_vectors[j],np.asscalar(weights[j]),d)

		print("gamma1: ",gamma[0])			
		for i in range(N):
			gamma[i,:] = gamma[i,:]/np.asscalar(gammatot[i])
		
		
		print("gamma2: ",gamma)
		print(np.max(gamma))
		# N_k = np.sum(gamma,axis=0)
		N_k = np.zeros((K,1))
		for j in range(K):
			for i in range(N):
				N_k[j] += gamma[i][j]

		for i in range(K):
			summ1 = 0
			for j in range(N):
				summ1 += np.asscalar(gamma[j][i])*X_input[:,j]

			weights[i] = N_k[i]/N
			mean_vectors[i] = np.divide(summ1,np.asscalar(N_k[i]))


		for i in range(K):
			summ2 = 0
			for j in range(N):
				diff = (X_input[:,j]-mean_vectors[i]).reshape((d,-1))
				summ2 += np.asscalar(gamma[j][i])*(np.dot(diff,diff.T))

			covariance_matrices[i] = np.divide(summ2,np.asscalar(N_k[i]))


	# # for i in range(N):
	# # 	temp = []
	# # 	for j in range(K):
	# # 		val = gmm(X_input[:,i],covariance_matrices[j],mean_vectors[j],weights[j],d)
	# # 		temp.append(val)
		
	# # 	tot = sum(temp)
	# # 	prob.append([temp[i]/tot for i in range(K)])

	# # prob = np.array(prob)
	# # m = np.sum(prob)

	# # for i in range(K):
	# # 	weights[i] = sum(prob[:,i])/m

	# # for i in range(K):
	# # 	summ = 0
	# # 	for j in range(N):
	# # 		summ += prob[j][i]*X_input[:,j]

	# # 	mean_vectors[i] = np.divide((summ*m),(weights[i]))

	# # for i in range(K):
	# # 	summ = 0
	# # 	for j in range(N):
	# # 		diff = np.array(X_input[:,j]-mean_vectors[i]).reshape((d,-1))
	# # 		summ += np.asscalar(prob[j][i])*(np.dot(diff,diff.T))

	# # 	covariance_matrices[i] = (summ*m)/np.asscalar(weights[i])

		loglikelihood_old = loglikelihood_new
		loglikelihood_new = loglikelihood(X_input,covariance_matrices,mean_vectors,weights,K,N,d)

		print("weights: \n",weights)
		print("mean_vectors: \n",mean_vectors)
		print("covariance_matrices: \n",covariance_matrices)
		print(loglikelihood_new,loglikelihood_old)
		print("\n\n\n\n\n\n")


	return weights,mean_vectors,covariance_matrices,gamma



if __name__ == '__main__':

	K = int(input("Enter the value of mixture size: "))
	X_input = readfile("data.txt")

	weights,mean_vectors,covariance_matrices,gamma = GMM(X_input,K)
	
	#print(gamma)
	labels=[]
	for g in gamma:
		labels.append(np.argmax(g))

	print(labels)
	plt.scatter(X_input[0], X_input[1], c=labels, s=40, cmap='viridis')
	plt.show()
