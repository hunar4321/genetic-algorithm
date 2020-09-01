"""
@author: Hunar @ Brainxyz
below is a very simple implementation of a GA based neural network which solves the XOR problem
"""
import numpy as np
import matplotlib.pyplot as plt

## net structure and initialization (single layer neural network)
inp = 2; hidden = 5; out = 1;
mutation_rate = 0.03
w1 = np.random.uniform(-1,1, (inp, hidden))
w2 = np.random.uniform(-1,1, (hidden, out))

def feedforward(X, w1, w2):
    z = np.tanh(X @ w1)
    yh = z @ w2
    return yh

def mutate(w, mutation_rate):
    ## mutate the weights by adding small random values to them
    dw = np.random.uniform(-1,1, (w.shape)) * mutation_rate
    m_w = w + dw
    return m_w
       
def train(X, y, w1, w2, iterations):
    errs = []
    for i in range(iterations):
        
        m_w1 = mutate(w1, mutation_rate) #mutate input-hidden weights
        m_w2 = mutate(w2, mutation_rate) #mutate hidden-output weights
        
        yh1 = feedforward(X,   w1,   w2); #parent network
        yh2 = feedforward(X, m_w1, m_w2); #child network (mutated weights)
        
        err_father = np.sum(np.abs(y - yh1.ravel())); #evaluate parent
        err_child =  np.sum(np.abs(y - yh2.ravel())); #evaluate child 
        
        if(err_child < err_father): # select better
            w1 = m_w1;
            w2 = m_w2;   
            
        errs.append(err_child)        
    return errs, w1, w2

## usage example: XOR data
X=np.asarray([[0,0],[1,1],[0,1],[1,0]]);
y=np.asarray([0,0,1,1]);

iterations = 1000 
errs, m_w1, m_w2 = train(X, y, w1, w2, iterations)

## check the results  
print(feedforward(X, m_w1, m_w2))      
plt.figure(1)
plt.plot(errs)
plt.title('errors')
