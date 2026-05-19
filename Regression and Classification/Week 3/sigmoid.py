"""
Description: Basic sigmoid function
Author: Aleksa Zatezalo
Date: May 19, 2026
"""

import numpy as np

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

print(sigmoid(1))
print(sigmoid_derivative(1))
print(sigmoid(10))
print(sigmoid_derivative(1))