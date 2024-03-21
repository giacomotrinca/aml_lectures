import numpy as np
import sys



def loss(x):
    return np.sum(x**2+2*x-6)

def derivative(x):
    return 2*x+2.

# GD parameters
epochs = int(sys.argv[1])   # epochs
eta = float(sys.argv[2])      # learning rate
dim = int(sys.argv[3])      # model dimensionality
tol = float(sys.argv[4])      # tolerance to break minimization

# Generate a random vector x in
w = np.random.rand(dim)
D = derivative(w)
L = loss(w)

cond = dim==2
# files
if cond:
    file_name="GD_loss_2d.dat"
    file = open(file_name, "w")    
for epoch in range(epochs):
    if cond:
        file.write(f'{epoch}\t{w[0]}\t{w[1]}\t{L}\t{D[0]}\t{D[1]}\t{np.linalg.norm(D)}\n')
        
    w_update = w - eta * D
    L_update = loss(w_update)
    D_update = derivative(w_update)
    print(f'{epoch}\t{w_update}\t{L_update}\t{D_update}')
    if np.linalg.norm(D_update)<tol:
        break
    else:
        w = w_update
        L = L_update
        D = D_update
    
