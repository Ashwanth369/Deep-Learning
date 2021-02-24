import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

Xmoon, ymoon = make_moons(400, noise=0.05, random_state=42)

plt.figure("Data Set",figsize=(100,100))
plt.scatter(Xmoon[:, 0], Xmoon[:, 1]);

np.savetxt("data.txt",Xmoon.T)
#plt.scatter(X_stretched[:,0], X_stretched[:,1], s=40)
plt.show()