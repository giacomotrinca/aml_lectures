import numpy as np

def loss(x):
    return np.sum(x**2)

def derivative(x):
    return 2*x

eta = 0.001
max_iter = 1000
w = np.random.rand(10)
D = derivative(w)
L = loss(w)
for _ in range(max_iter):
    w_update = w - eta * D
    L_update = loss(w_update)
    
    

    
