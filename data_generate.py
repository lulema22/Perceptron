import numpy as np
N = 1000000
l = 30

X = np.random.rand(N, l)
np.random.seed(45)
w = np.random.uniform(-4, 4, size=30)
bios = -1.0 # попробовать поменять
z = X @ w + bios + np.random.normal(0, 0.5, size=1_000_000)# c шумом еще сделать


y = (z > 0).astype(int)

np.save('X.npy', X)
np.save('y.npy', y)

