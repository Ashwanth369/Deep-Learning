import numpy as np
import random

def readfile(fname):
	data = []
	f1 = open(fname,"r")
	f2 = f1.readlines()
	
	for line in f2:
		row = list(map(float,line.split()))
		data.append(row)

	data = np.array(data)
	return data

a=readfile("data.txt")
d,N = a.shape[0],a.shape[1]
# d = int(input("Enter the number of rows of the observation matrix: "))
# N = int(input("Enter the number of columns of the observation matrix: "))
K = int(input("Enter the mixture size: "))
# print("Enter all inputs in order of rows: ")
# a=np.zeros((d,N))
# for i in range(d):
# 	nums = list(map(float,raw_input().split()))
# 	for j in range(N):
# 		a[i][j] =nums[j]
# a=np.asarray(a)

def gaussian(a,u,sigma):
	return (1/(np.sqrt(((2*np.pi)**d)*abs(np.linalg.det(sigma)))))*np.exp((-0.5)*np.dot((a-u).transpose(),np.dot(np.linalg.inv(sigma),(a-u))))
# def gamma():

def check(loglike,logliken):
	if(abs(loglike-logliken)<0.01):
		return 0
	else:
		return 1

pi_k=np.zeros((K,1))
for i in range(K):
	pi_k[i,0]=1.0/K 
u_k=np.zeros((d,K))
for i in range(K):
	for j in range(d):
		u_k[j,i]=0
# u_k=np.random.rand(d,K)
sigma_k=(np.random.rand(K,d,d))*500
gamma=np.zeros((N,K))
gammatotal=np.zeros((N,1))

print(sigma_k)

loglike=0
for i in range(N):
	for j in range(K):
		loglike+=pi_k[j,0]*gaussian(a[:,i].reshape(d,1),u_k[:,j].reshape(d,1),sigma_k[j,:,:])
# print("===")
# print(loglike)

while(1):
	print("///")
	print(loglike)
	gamma=np.zeros((N,K))
	gammatotal=np.zeros((N,1))
	for i in range(N):
		for j in range(K):
			gammatotal[i,0]+=pi_k[j,0]*gaussian(a[:,i].reshape(d,1),u_k[:,j].reshape(d,1),sigma_k[j,:,:])
	# print(gammatotal)

	for i in range(N):
		for j in range(K):
			gamma[i,j]=pi_k[j,0]*gaussian(a[:,i].reshape(d,1),u_k[:,j].reshape(d,1),sigma_k[j,:,:])
		gamma[i,:]=gamma[i,:]/gammatotal[i,0]
	
	print("gamma: ",gamma)
	print(np.max(gamma))

	Nk=np.zeros((K,1))
	for j in range(K):
		for i in range(N):
			Nk[j,0]+=gamma[i,j]

	print("Nk: ",Nk)
	u_kn=np.zeros((d,K))
	for i in range(K):
		for j in range(N):
			u_kn[:,i]+=gamma[j,i]*a[:,j]
		u_kn[:,i]=u_kn[:,i]/Nk[i,0]

	print("u_kn: ",u_kn)

	sigma_kn=np.zeros((K,d,d))
	for i in range(K):
		for j in range(N):
			sigma_kn[i,:,:]+=gamma[j,i]*np.dot(a[:,j].reshape(d,1)-u_kn[:,i].reshape(d,1),(a[:,j].reshape(d,1)-u_kn[:,i].reshape(d,1)).transpose())
		sigma_kn[i,:,:]=sigma_kn[i,:,:]/Nk[i,0]

	print("sigma_kn: ",sigma_kn)
	pi_kn=np.zeros((K,1))
	for i in range(K):
		pi_kn[i,0]=Nk[i,0]/N

	# print(u_kn,sigma_kn,pi_kn)
	logliken=0
	for i in range(N):
		per=0
		for j in range(K):
			per+=pi_kn[j,0]*gaussian(a[:,i].reshape(d,1),u_kn[:,j].reshape(d,1),sigma_kn[j,:,:])
		per=np.log(per)
		logliken+=per
	print("===")
	print(logliken)
	print("\n")
	# print("****")
	# print(sigma_kn)
	if(check(np.linalg.det(sigma_k[0,:,:]),np.linalg.det(sigma_kn[0,:,:]))==0):	
		print(u_kn)
		print(sigma_kn)
		print(pi_kn)
		break
	else:
		u_k=u_kn
		sigma_k=sigma_kn
		pi_k=pi_kn
		loglike=logliken
# print(a[:,0].reshape(d,1),u_k[:,0].reshape(d,1),sigma_k[0,:,:])
# print(gaussian(a[:,1].reshape(d,1),u_k[0,:].reshape(d,1),sigma_k[0,:,:]))
# print(pi_k)
# print(u_k)
# print(sigma_k)

# print(gaussian(a[:,1].reshape(d,1),u_k[0,:].reshape(d,1),sigma_k[0,:,:]))



