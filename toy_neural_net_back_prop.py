# -*- coding: utf-8 -*-
"""

@author: Iamtrask
URL: http://iamtrask.github.io/2015/07/12/basic-python-network/

Modified by Andrew
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x,deriv=False):
    if (deriv==True):
        #return x*(1-x) #From original code
        return np.exp(x) / ((np.exp(x) + 1)**2)
    return 1 / (1+np.exp(-x))
    
# Input dataset
X = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])

# Output dataset
y = np.array([[0,0,1,1]]).T

# Seed
np.random.seed(1)

# Initialize weights randomly with mean 0
synapse01 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    
    # Forward propogation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0,synapse01))
    
    # Error
    err = y - layer1
    
    # Multiply error by slope of sigmoid at layer1
    layer1_delta = err * sigmoid(layer1,True)
    
    #pdb.set_trace()
    # Update weights
    synapse01 += np.dot(layer0.T,layer1_delta)
    
print("Output after training:")
print(layer1)