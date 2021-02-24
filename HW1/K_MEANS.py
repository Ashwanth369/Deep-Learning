#########################                 IMPORTANT                 ##############################

#For giving an input, please provide the data in the form of dxN in the data.txt file.
#I am uploading an image in the zip file which will return a clustered image.



import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def colour():
	return '#%02x%02x%02x' % (random.randint(0,255),random.randint(0,255),random.randint(0,255))

def readimage(fname):
	img = cv2.imread(fname)
	R,G,B = cv2.split(img)
	R = R.flatten()
	G = G.flatten()
	B = B.flatten()

	data = [[R],[G],[B]]
	data = np.array(data,dtype='int64').reshape((3,img.shape[0]*img.shape[1]))
	return data,img.shape[0],img.shape[1]


def saveimage(labels,centroids,res1,res2):
	N = labels.shape[0]
	img = []
	for i in range(N):
		img.append(centroids[labels[i]])
	img = np.array(img)
	img = np.uint8(np.lib.stride_tricks.as_strided(img,(res1,res2,3)))
	cv2.imwrite('Clustered_Image.jpg',img)


def readfile(fname):
	data = []
	f1 = open(fname,"r")
	f2 = f1.readlines()
	
	for line in f2:
		row = list(map(float,line.split()))
		data.append(row)

	data = np.array(data)
	return data


def plot_graph(X_input,labels,centroids,K):
	
	d,N = X_input.shape
	
	if d == 2:
		plt.figure(figsize=(100,100))
		plt.subplot(1,2,1).set_title("Input KMeans")
		plt.scatter(X_input[0,:],X_input[1,:])

		plt.subplot(1,2,2).set_title("Output KMeans")
		for i in range(K):
			c = colour()
			for j in range(N):
				if labels[j] == i:
					plt.scatter(X_input[0,j],X_input[1,j],color=c)
			plt.scatter(centroids[i][0],centroids[i][1],color=c,marker='*')

		plt.show()

	if d == 3:
		fig1 = plt.figure()
		ax1 = Axes3D(fig1)
		ax1.scatter(X_input[0,:],X_input[1,:],X_input[2,:])

		fig2 = plt.figure()
		ax2 = Axes3D(fig2)

		for i in range(K):
			c = colour()
			for j in range(N):
				if labels[j] == i:
		 			ax2.scatter(X_input[0,j],X_input[1,j],X_input[2,j],color=c)
			ax2.scatter(centroids[i][0],centroids[i][1],centroids[i][2],color=c,marker='*')
		plt.show()


class KMeans:

	labels_ = np.array([])
	centroids_ = np.array([])

	def __init__(self,n_clusters=4,threshold=0.000001):
		self.n_clusters = n_clusters
		self.threshold = threshold

	def centers(self,X_input):
		d,N = X_input.shape
		
		centroids = []
		points = [np.random.randint(0,N)]
		centroids.append(X_input[:,points[0]])

		for i in range(self.n_clusters-1):
			Norms = []

			for j in range(N):
				if j in points:
					Norms.append(0)
				else:
					Norms.append(np.linalg.norm(X_input[:,j]-centroids[i]))

			TotDist = sum(Norms)
			Prob = [x/TotDist for x in Norms]
			k = np.asscalar(np.random.choice(N,1,p=Prob))
			centroids.append(X_input[:,k])
			points.append(k)

		return np.array(centroids)

	def fit(self,X_input):

		d,N = X_input.shape

		if N < self.n_clusters:
			warnings.warn("Number of input data points ({}) found less than Number of clusters ({}).".format(N,self.n_clusters))

		centroids_old = self.centers(X_input)

		centroids = np.zeros(self.n_clusters)

		error = 100
		cluster_labels = np.zeros(N)

		iteration = 1

		while error > self.threshold:

			for i in range(N):
				MinValue = 1e10
				index = 0

				for j in range(self.n_clusters):
					if(np.linalg.norm(X_input[:,i]-centroids_old[j]) < MinValue):
						MinValue = np.linalg.norm(X_input[:,i]-centroids_old[j])
						index = j

				cluster_labels[i] = index

			centroids = np.copy(centroids_old)

			for i in range(self.n_clusters):
				points = [X_input[:,j] for j in range(N) if cluster_labels[j] == i]
				centroids_old[i] = np.mean(points,axis=0)

			error = np.linalg.norm(centroids-centroids_old)

			#print("---------  Iteration ({}) ------ Error: ({})" .format(iteration,error))
			iteration += 1

		self.labels_ = cluster_labels.astype(int)
		self.centroids_ = centroids
		return self


if __name__ == '__main__':

	Flag = int(input("Enter 1 if you want to input to check how the image is clustered, else 0: "))
	K = int(input("Enter the number of clusters: "))
	epsilon = float(input("Enter the value of epsilon: "))

	if Flag == 0:
		X_input = readfile("data.txt")
	elif Flag == 1:
		X_input,res1,res2 = readimage("img.jpg")
		
	Y_output = KMeans(n_clusters=K,threshold=epsilon).fit(X_input)

	if Flag == 1 :
		saveimage(Y_output.labels_,Y_output.centroids_,res1,res2)


	print("K Centroids at Convergence:\n",Y_output.centroids_)
	print("\n")

	for i in range(K):
		print("Cluster "+str(i+1)+" : ")
		A = []
		for j in range(Y_output.labels_.shape[0]):
			if Y_output.labels_[j] == i:
				A.append(X_input[:,j])
		print(np.array(A))
		print("\n")

	#Uncomment the following line for a 2D,3D graph plot of the clusters if possible.
	#plot_graph(X_input,Y_output.labels_,Y_output.centroids_,K)