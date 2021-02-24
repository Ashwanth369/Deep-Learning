import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidDer(x):
	return np.exp(-x)/(1+np.exp(-x))**2

def test_train_split(data,labels,n,ratio):
	x_train,x_test,y_train,y_test = [],[],[],[]
	temp = n*ratio
	
	for i in range(n):
		if np.random.randint(0,n) < temp:
			x_train.append(data[i])
			y_train.append(labels[i])
		else:
			x_test.append(data[i])
			y_test.append(labels[i])

	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)


class MLP:

	def __init__(self,n_layers_sizes,learning_rate=0.01,num_epochs=20000):

		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.n_layers = len(n_layers_sizes) 
		self.biases = np.random.normal(0,0.1,self.n_layers-1)
		self.weights = [np.random.normal(0,0.1,(i,j)) for i,j in zip(n_layers_sizes[:-1],n_layers_sizes[1:])]


	def fit(self,X):

		i = 0
		batch_size = 50
		batch_index = batch_size
		flag = 0

		while i < self.num_epochs:

			if batch_index >= X.shape[0]:
				batch_index = X.shape[0]
				flag = 1

			z_s,a_s = self.forwardandbackprop(X[batch_index-batch_size:batch_size])
			der_b,der_w,e = self.backprop(a_s,z_s,X[batch_index-batch_size:batch_size])

			for j in range(self.n_layers-1):
				self.biases[j] -= self.learning_rate*der_b[j]
				self.weights[j] -= self.learning_rate*der_w[j]

			print("EPOCH "+str(i+1)+" ---- ERROR "+str(e))
			
			if flag == 1:
				batch_index = batch_size

			i = i + 1

		return self


	def predict(self,x):
		
		a_s = [x]
		z_s = []

		for b,w in zip(self.biases,self.weights):
			z = np.dot(a_s[-1],w)+b
			z_s.append(z)
			a_s.append(sigmoid(z))

		return a_s[-1]		


	def forwardandbackprop(self,x):

		a_s = [x]
		z_s = []

		for b,w in zip(self.biases,self.weights):
			z = np.dot(a_s[-1],w)+b
			z_s.append(z)
			a_s.append(sigmoid(z))

		return z_s,a_s

	def backprop(self,a_s,z_s,y):

		der_b = np.zeros(self.n_layers-1)
		der_w = [np.zeros(w.shape) for w in self.weights]

		delta = np.multiply((a_s[-1]-y),sigmoidDer(z_s[-1]))

		der_b[-1] = np.sum(delta)
		
		der_w[-1] = np.dot(a_s[-2].T,delta)

		for i in range(2,self.n_layers):

			delta = np.multiply(np.dot(delta,self.weights[-i+1].T),sigmoidDer(z_s[-i]))
			der_w[-i] = np.dot(a_s[-i-1].T,delta)
			der_b[-i] = np.sum(delta)

		return der_b,der_w,np.linalg.norm(a_s[-1][0]-y[0])


if __name__ == '__main__':

	digits = datasets.load_digits()
	images = digits.images
	labels = digits.target
	images /= 16.0

	x_train,y_train,x_test,y_test = test_train_split(images,labels,len(labels),0.75)
	x_train = x_train.reshape(len(x_train),-1)
	
	n_layers_sizes = [64,30,64]
	mlp = MLP(n_layers_sizes).fit(x_train)
	

	#Visualization of prediction on test data.
	t = []
	p = []
	fig,ax = plt.subplots(2,5)
	ax = ax.flatten()
	for i in range(10):
		if i < 5 :
			t.append(x_test[np.random.randint(0,len(x_test))].reshape(64))
			ax[i].imshow(t[i].reshape(8,8),cmap="Greys")
		else:
			p.append(mlp.predict(t[i-5]))
			ax[i].imshow(p[i-5].reshape(8,8),cmap="Greys")

	plt.show()