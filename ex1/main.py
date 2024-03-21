#Ex1: A simple MLP
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split


import torch
import torchvision

print("Torch: \t\t", torch.__version__)
print("TorchVision:\t",torchvision.__version__)

if torch.cuda.is_available():
    print(f'Number of available devices: {torch.cuda.device_count()}')
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())   
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")
    
features_S = np.random.normal(loc=[-4.,-3.,-2.,-1.,1.,2.,3.,4.], scale=np.random.uniform(0.2,0.7,size=(8)), size=(2000,8))
features_B = np.random.normal(loc=[-4.5,-3.5,-2.5,-1.5,1.5,2.5,3.5,4.5], scale=np.random.uniform(0.2,0.7,size=(8)), size=(2000,8))
labels_S = np.zeros(shape=(2000))
labels_B = np.ones(shape=(2000))

X = np.concatenate((features_S, features_B), axis=0)
Y = np.concatenate((labels_S, labels_B), axis=0)
    
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=12345)
X_train, X_vali, Y_train, Y_vali = train_test_split(X_train, Y_train, test_size=0.2, random_state=456789)
print(X_train.shape)
print(Y_train.shape)
print(X_vali.shape)
print(Y_vali.shape)
print(X_test.shape)
print(Y_test.shape)

sel_S = tuple([Y_train<0.5])
sel_B = tuple([Y_train>0.5])

for idx in range(8):
  plt.subplot(2,4,idx+1)
  plt.hist((X_train[sel_S])[:,idx], bins=50, range=[-6.,6.], alpha=0.5, log=True, density=True)
  plt.hist((X_train[sel_B])[:,idx], bins=50, range=[-6.,6.], alpha=0.5, log=True, density=True)
plt.tight_layout()