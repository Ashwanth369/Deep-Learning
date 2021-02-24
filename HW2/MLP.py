#################################  NOTES  #################################

# The number of hidden layers and number of nodes in each layer can be changed.
# Total number of training samples is 10000*4 = 40000 with some noice.
# Parameters: Number of epochs = 10000, Learning rate = 0.1
# The number of layers in the hidden layer and also number of nodes in each hidden layer is configurable.


import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidDer(x):
	return np.exp(-x)/(1+np.exp(-x))**2

def gen(k):
	if k==1:
		return np.array([[0],[1]])
	else:
		c = gen(k-1)
		a = np.hstack((np.zeros((2**(k-1),1)),c))
		b = np.hstack((np.ones((2**(k-1),1)),c))
		z = np.vstack((a,b))

		return z;


class MLP:

	def __init__(self,n_layers_sizes):

		self.n_layers = len(n_layers_sizes) 
		self.biases = np.random.rand(self.n_layers-1)
		self.weights = [np.random.rand(i,j) for i,j in zip(n_layers_sizes[:-1],n_layers_sizes[1:])]


	def fit(self,flag):

		error = 100
		i = 0
		
		while i < 10000:
			
			#### INPUT DATA AND OUTPUT LABELS WITH SOME NOISE.
			X = np.array([[np.random.uniform(-0.01,0.01),np.random.uniform(-0.01,0.01)],[1+np.random.uniform(-0.01,0.01),np.random.uniform(-0.01,0.01)],[np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01)],[1+np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01)]])
			
			Y = []

			if flag == 1:
				Y = [np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01),np.random.uniform(-0.01,0.01)]

			elif flag == 2:
				Y = [np.random.uniform(-0.01,0.01),np.random.uniform(-0.01,0.01),np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01)]

			elif flag == 3:
				Y = [np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01),1+np.random.uniform(-0.01,0.01)]

			Y = np.array(Y)


			##PERFORM FORWARD AND BACKPROP.UPDATE WEIGHTS AND BIASES.
			for n in range(X.shape[0]):
				
				z_s,a_s = self.forward(X[n])
				der_b,der_w,e = self.backprop(a_s,z_s,Y[n])

				for j in range(self.n_layers-1):
					self.biases[j] -= 0.1*der_b[j]
					self.weights[j] -= 0.1*der_w[j]

			
			error = e
			print("EPOCH "+str(i+1)+" ---- ERROR "+str(error))
			i = i + 1


		return self


	def predict(self,x):

		z_s,a_s = self.forward(x)
		
		if a_s[-1] > 0.5:
			return 1
		else:
			return 0
			

	def forward(self,x):

		a_s = [x]
		z_s = []

		for b,w in zip(self.biases,self.weights):
			z = a_s[-1].dot(w)+b
			z_s.append(z)
			a_s.append(sigmoid(z))

		return z_s,a_s


	def backprop(self,a_s,z_s,y):

		der_b = np.zeros(self.n_layers-1)
		der_w = [np.zeros(w.shape) for w in self.weights]

		delta = 2*(a_s[-1]-y)*sigmoidDer(z_s[-1])

		der_b[-1] = np.sum(delta)

		for j in range(der_w[-1].shape[1]):
			for k in range(der_w[-1].shape[0]):
				der_w[-1][k][j] = a_s[-2][k]*delta[j]

		for i in range(2,self.n_layers):

			delta2 = np.zeros(z_s[-i].shape[0])
			for j in range(z_s[-i].shape[0]):
				for m in range(z_s[-i+1].shape[0]):
					delta2[j] += self.weights[-i+1][j][m]*delta[m]

			delta = delta2*sigmoidDer(z_s[-i])

			for j in range(der_w[-i].shape[1]):
				for k in range(der_w[-i].shape[0]):
					der_w[-i][k][j] = a_s[-i-1][k]*delta[j]

			der_b[-i] = np.sum(delta)

		return der_b,der_w,cost




if __name__ == '__main__':
		
	Flag = int(input("XOR:1\tAND:2\tOR:3\nEnter the corresponding value: "))

	test_y = []

	if Flag == 1:
		test_y = [0,1,1,0]

	elif Flag == 2:
		test_y = [0,0,0,1]

	elif Flag == 3:
		test_y = [0,1,1,1]


	n_layers_sizes = [2,2,1]
	mlp = MLP(n_layers_sizes).fit(Flag)

	test_x = np.array([[0,0],[1,0],[0,1],[1,1]])
	test_y = np.array(test_y)
	predicted_y = []
	accuracy = 0

	for n in range(test_x.shape[0]):
		y = mlp.predict(test_x[n])
		predicted_y.append(y)
		if y == test_y[n]:
			accuracy += 1

	predicted_y = np.array(predicted_y)
	accuracy /= test_x.shape[0]
	
	print("\n\n")
	for n in range(test_x.shape[0]):
		print("Input: ")
		print(test_x[n])
		print("Output: ")
		print(test_y[n])
		print("Predicted: ")
		print(predicted_y[n])
		print("\n\n")

	print("Accuracy: "+str(accuracy*100)+"%")
