#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:36:18 2018

@author: ben
"""

from astropy.io import fits
from scipy.optimize import nnls 

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-white') 

import numpy as np
import scipy as sci

A = fits.getdata('Q_tomo.fits')

print(np.linalg.cond(A))
mu = 1.0/np.linalg.norm(A)**2
print(mu)

#plt.figure()
#plt.imshow(A,cmap='magma',interpolation='nearest')

x_true = np.zeros(50)
x_true[21] = 1.
b = np.dot(A,x_true)

#b += np.random.standard_normal(b.shape) * 1e-6

plt.figure('True solution')
plt.plot(x_true)


plt.figure('Given right hand side')
plt.plot(b)



#===================================
# Type of regularization
#===================================
reg = 'l0' # or 'l0'

#===================================
# Set tuning parameters
#===================================
_, s, _ = sci.linalg.svd(A, False)
kappa = s[0] ** 2 * 0.001
nu = 1 / kappa * 0.1
lamb = 1e-11 / 3 # controll level of sparsity 
beta = 1e-12 # add some ridge


maxiter = 20000 # max iterations

#===================================
# Init Algorithm
#===================================
C = np.eye(len(x_true))
H = A.T.dot(A) + kappa * C.T.dot(C)
Hinv = sci.linalg.pinv2(H)

Fupper = kappa * A.dot(Hinv).dot(C.T)
Flower = np.sqrt(kappa) * (np.eye(C.shape[0]) - kappa * C.dot(Hinv).dot(C.T))
F = np.concatenate((Fupper, Flower))

Gupper = np.eye(A.shape[0]) - A.dot(Hinv).dot(A.T)
Glower = np.sqrt(kappa) * C.dot(Hinv).dot(A.T)
G = np.concatenate((Gupper, Glower))

g = G.dot(b)

#===================================
# Project
#===================================
Q , _, _ = sci.linalg.svd(F, False)

Q = Q[:,0:5]

F = Q.T.dot(F)
g = Q.T.dot(g)

w = np.zeros(x_true.shape)
for i in range(maxiter):

        grad = F.T.dot(F.dot(w) - g) - beta * w
        w_temp = w - nu * grad

        if reg == 'l1':
            # l1 soft-threshold
            idxH = w_temp > lamb * nu
            idxL = w_temp <= -lamb * nu
            w = np.zeros_like(w_temp)
            w[idxH] = w_temp[idxH] - lamb * nu
            w[idxL] = w_temp[idxL] + lamb * nu

        if reg == 'l0':

            # l0 soft-threshold
            idxH = w_temp**2 > (lamb * nu) ** 2
            w = np.zeros_like(w_temp)
            w[idxH] = w_temp[idxH]        
            w = np.maximum(w, 0.0)
        
  
plt.figure('Estimate for w')
plt.plot(w)
      
xapprox = Hinv.dot(A.T.dot(b) + kappa * C.T.dot(w))
        
plt.figure('Estimate for x')
plt.plot(xapprox)
