# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:39:48 2017

@author: cmh
"""
import numpy as np
import matplotlib.pyplot as plt
# assume some unit gaussian 10-D input data
D = np.random.randn(1000, 500)
hidden_layer_sizes = [500]*10
nonlinearities = ['tanh']*len(hidden_layer_sizes)
#%%
act = {'relu':lambda x:np.maximum(0,x),'tanh':lambda x:np.tanh(x)}
Hs = {}
for i in range(len(hidden_layer_sizes)):
    X = D if i==0 else Hs[i-1] #input at thie layer
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in,fan_out)*0.05 #layer initialization

    H = np.dot(X,W) #matrix mutiply
    H = act[nonlinearities[i]](H) #nonlinearity
    Hs[i] = H #cache result on this year
#%%
print('input layer had mean %f and std %f' % (np.mean(D),np.std(D)))
layer_means = [np.mean(H) for i,H in Hs.items()]
layer_stds = [np.std(H)  for i,H in Hs.items()]
for i,H in Hs.items():
    print ('hidden layder %d had mena %f and std %f'% (i+1,layer_means[i],layer_stds[i]))

#%%
a = list(Hs.keys())
x = np.array(a).reshape(10,1)
y = np.array(layer_means).reshape(10,1)
z = np.array(layer_stds).reshape(10,1)
plt.figure()
plt.subplot(121)
plt.plot(x,y,'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(x,z,'or-')
plt.title('layer std')

#%%
plt.figure()
for i,H in Hs.items():
    plt.subplot(1,len(Hs),i+1)
    plt.hist(H.ravel(),30,range=(-1,1))
