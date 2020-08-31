"""
@author: Hunar @ Brainxyz
below is a very simple implementation of a GA based neural network which solves the XOR problem
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

def feedforward(X, w1, w2):
    z = np.tanh(X @ w1)
    yh = z @ w2
    return yh

errs = []; # to store errors and track performance
for i in range(1000):
    
    ## mutate the weights by adding small random values
    dw1 = np.random.uniform(-1,1, (w1.shape))
    dw2 = np.random.uniform(-1,1, (w2.shape))
    m_w1 = w1 + dw1 * mutation_rate
    m_w2 = w2 + dw2 * mutation_rate
    
    ## feedforward1 - father network
    yh = feedforward(X, w1, w2);
    err_father = np.sum(np.abs(y - yh.ravel()));

    ## feedforward2 - child network (mutated weights)
    yh = feedforward(X, m_w1, m_w2);
    err_child = np.sum(np.abs(y - yh.ravel()));
    
    ## if the child net performance is better than father then update weights
    if(err_child < err_father):
        w1 = m_w1;
        w2 = m_w2;
    errs.append(err_child)
## check the results        
plt.figure(1)
plt.plot(errs)
plt.title('err')
print(feedforward(X, w1, w2))
    
