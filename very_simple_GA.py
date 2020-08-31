# -*- coding: utf-8 -*-
"""
@author: Hunar @ Brainxyz
"""
import numpy as np
import matplotlib.pyplot as plt

## XOR data
X=np.asarray([[0,0],[1,1],[0,1],[1,0]]);
y=np.asarray([0,0,1,1]);

## net structure and initialization
inp = 2; hidden = 5; out = 1;
mutation_rate = 0.03
w1 = np.random.uniform(-1,1, (inp, hidden))
w2 = np.random.uniform(-1,1, (hidden, out))

## feedforward 
def feedforward(X, w1, w2):
    z = np.tanh(X @ w1)
    yh = z @ w2
    return yh

errs = []; # to store errs and track perforamnce
for i in range(1000):
    
    ## mutate the weights
    m_w1 = w1 + np.random.uniform(-1,1, (w1.shape)) * mutation_rate
    m_w2 = w2 + np.random.uniform(-1,1, (w2.shape)) * mutation_rate
    
    ## feedforward1 - father network
    yh = feedforward(X, w1, w2);
    err_father = np.sum(np.abs(y - yh.ravel()));

    ## feedforward2 - child network
    yh = feedforward(X, m_w1, m_w2);
    err_child = np.sum(np.abs(y - yh.ravel()));
    
    ## if child net performance better than father then update
    if(err_child < err_father):
        w1 = m_w1;
        w2 = m_w2;
        
    errs.append(err_child)
        
plt.figure(1)
plt.plot(errs)
plt.title('err')

## check 
print(feedforward(X, w1, w2))
        
    