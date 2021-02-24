#########################                 IMPORTANT                 ##############################

#For giving an input, please provide the data in the form of dxN in the data.txt file.
#I am uploading an image in the zip file which will return a image after applying PCA.

import numpy as np
import matplotlib.pyplot as plt
import cv2

def readimage(fname):
	img = cv2.imread(fname)
	R,G,B = cv2.split(img)
	R = R.flatten()
	G = G.flatten()
	B = B.flatten()

	data = [[R],[G],[B]]
	data = np.array(data,dtype='int64').reshape((3,img.shape[0]*img.shape[1]))
	return data,img.shape[0],img.shape[1]

def saveimage(Y_output,res1,res2):
	img = Y_output.T.reshape((res1,res2,3))
	cv2.imwrite('PCA_Image.jpg',img)


def readfile(fname):
	data = []
	f1 = open(fname,"r")
	f2 = f1.readlines()
	
	for line in f2:
		row = list(map(float,line.split()))
		data.append(row)

	data = np.array(data)

	return data

def plot_graph(X_input,Y_output):
	plt.figure(figsize=(100,100))
	plt.subplot(1,2,1).set_title("Input PCA")
	plt.scatter(X_input[0,:],X_input[1,:])
	plt.subplot(1,2,2).set_title("Output PCA")
	plt.scatter(Y_output[0,:],Y_output[1,:])
	plt.xlim(-30,30)
	plt.ylim(-30,30)
	plt.show()

def PCA(X_input):

	d,N = X_input.shape
	mean = np.mean(X_input,axis=1).reshape((d,-1))
	X_input = X_input - mean
	covariance = np.cov(X_input)
	print("\nThe covariance matrix of input,X:")
	print(covariance)

	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	print("\nEigen Vectors:")
	print(eigenvectors)


	out = eigenvectors.T.dot(X_input)
	return out


if __name__ == '__main__':

	Flag = int(input("Enter 1 if you want to input to check how the image is clustered, else 0: "))
	
	if Flag == 0:
		X_input = readfile("data.txt")
	elif Flag == 1:
		X_input,res1,res2 = readimage("img.jpg")
	
	Y_output = PCA(X_input)
	
	print("\nThe covariance matrix of output,Y:")
	print(np.cov(Y_output))

	if Flag == 1 :
		saveimage(Y_output,res1,res2)
	
	#Uncomment the below line for a 2d graph plot if possible.
	#plot_graph(X_input,Y_output)