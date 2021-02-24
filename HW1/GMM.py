#########################                 IMPORTANT                 ##############################
#---- PLEASE PROVIDE THE DATA IN THE FORM OF dxN IN THE data.txt FILE ----#
#---- THE SIZE OF THE POINTS IN THE PLOT OF GMM INDICATE THE PROBABILITY WITH WHICH THEY FALL INTO THAT MIXTURE ----#


import numpy as np
import random
import matplotlib.pyplot as plt
import K_MEANS

def readfile(fname):
	data = []
	f1 = open(fname,"r")
	f2 = f1.readlines()
	
	for line in f2:
		row = list(map(float,line.split()))
		data.append(row)

	data = np.array(data)
	return data


class GMM:
	
	probs_ = np.array([])
	pi_k_ = np.array([])
	mean_k_ = np.array([])
	cov_k_ = np.array([])

	

	def __init__(self,n_mixtures=3,threshold=0.001):
		self.n_mixtures = n_mixtures
		self.threshold = threshold

	def gaussian(self,x,mean,cov,d):
	
		i=0
		while np.linalg.det(cov)<0.000000001 and i<cov.shape[0]: 
			cov[i][i] += cov[i][i]/100
			i +=1

		return np.exp(-0.5*(np.dot((x-mean).reshape((d,-1)).T,np.dot(np.linalg.inv(cov),(x-mean).reshape((d,-1))))))/(np.sqrt(((2*np.pi)**d)*abs(np.linalg.det(cov))))


	def kmeans_initialization(self,X_input):

		d,N = X_input.shape
		K = self.n_mixtures

		Y_output = K_MEANS.KMeans(n_clusters=K).fit(X_input)
		
		pi_k = np.empty(K)
		mean_k = np.empty((K,d))
		cov_k = np.empty((K,d,d))
		
		for i in range(K):
		
			temp = []
			count = 0
		
			for j in range(N):
				if Y_output.labels_[j] == i:
					count += 1
					temp.append(X_input[:,j])

			temp = np.array(temp)
			
			pi_k[i] = count/N
			mean_k[i] = np.sum(temp,axis=0)/count
			cov_k[i] = np.cov(temp.T)
		
		if d==2:

			plt.figure("K Means",figsize=(100,100))
			plt.scatter(X_input[0], X_input[1], c=Y_output.labels_, s=60, cmap='gist_rainbow')

		
		return pi_k,mean_k,cov_k


	def loglikelihood(self,X_input,pi_k,mean_k,cov_k):

		d,N = X_input.shape
		K = self.n_mixtures
  
		ll = 0
		for n in range(N):
			temp = 0
			for k in range(K):
			  	temp += pi_k[k]*self.gaussian(X_input[:,n],mean_k[k],cov_k[k],d)

			ll += np.log(temp)

		return ll
		

	def fit(self,X_input):

		d,N = X_input.shape[0],X_input.shape[1]
		
		pi_k,mean_k,cov_k = self.kmeans_initialization(X_input)
		
		loglikelihood_new = self.loglikelihood(X_input,pi_k,mean_k,cov_k)
		
		gamma = np.zeros((N,K))
		stop_flag = False

		while not stop_flag:

			for n in range(N):
				
				for k in range(K):
					gamma[n,k] = pi_k[k]*self.gaussian(X_input[:,n],mean_k[k],cov_k[k],d)

				gamma_n = np.sum(gamma[n])
				gamma[n,:] = gamma[n,:]/gamma_n


			N_k = np.zeros(K)
			
			for k in range(K):
				for n in range(N):
					N_k[k] += gamma[n,k]


			for k in range(K):
				for n in range(N):
					mean_k[k] += gamma[n,k]*X_input[:,n]

				mean_k[k] = mean_k[k]/N_k[k]
				pi_k[k] = N_k[k]/N


			for k in range(K):
				for n in range(N):
					cov_k[k] += gamma[n,k]*np.dot((X_input[:,n]-mean_k[k]).reshape((d,-1)),(X_input[:,n]-mean_k[k]).reshape((d,-1)).T)

				cov_k[k] = cov_k[k]/N_k[k]


			loglikelihood_old = loglikelihood_new
			loglikelihood_new = self.loglikelihood(X_input,pi_k,mean_k,cov_k)

			
			if abs(loglikelihood_new - loglikelihood_old) < self.threshold:
				stop_flag = True


		self.pi_k_ = pi_k
		self.mean_k_ = mean_k
		self.cov_k_ = cov_k
		self.probs_ = gamma
		return self



if __name__ == '__main__':

	print("---- PLEASE PROVIDE THE DATA IN THE data.txt FILE ----")

	K = int(input("Enter the value of mixture size: "))
	X_input = readfile("data.txt")

	gmm = GMM(n_mixtures=K).fit(X_input)

	print("\n\n")
	print("Weights: \n",gmm.pi_k_)
	print("\n\n")
	print("Mean Vectors: \n",gmm.mean_k_)
	print("\n\n")
	print("Covariance Matrices: \n",gmm.cov_k_)
	print("\n\n")

	if X_input.shape[0]==2:
		
		labels = []
		
		for p in gmm.probs_:
			labels.append(np.argmax(p))

		size = [60*(gmm.probs_[n,labels[n]]**2) for n in range(X_input.shape[1])]

		plt.figure("GMM",figsize=(100,100))
		plt.scatter(X_input[0], X_input[1], c=labels, s=size, cmap='gist_rainbow')

		Flag = int(input("An input dimension of 2 is found, Enter 1 to plot the data, else 0: "))
		if Flag == 1:
			plt.show()
