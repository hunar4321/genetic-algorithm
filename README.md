# Genetic Algorthim
A simple evolutionary-genetic algorithm (Simultaneous update of all the weights) implemented both in python and C++.
The algorithm is very simple conceptually. it competes between two networks: father (f) & child (c) network and simultaneously mutates all the weights of the child network then chooses the better network to become the father of the next generation. For low dimensional data, this algorithm can be as efficient as naive gradient decent and that is because it doesn't require the backward pass (back-propagtion), instead it requires 2 forward passes (which are less intensive computationally than the backward pass). Also, if the loss function is not convex or the local minima hides in a narrow and elongated ridge, then this algorithm can be more efficient than the naive gradient decent which produces a zigzagged path towards the minima.

link to google colab (python):

https://colab.research.google.com/drive/18Q9W9WcjnWt8sVcWRIORb1KWrRry6-5L


**Disadvantages of genetic algorthim:**
1. It is a population based learning (can be inefficient without parallelization).
2. If mutation rate is high the learning is unpredictable and noisy. If mutation rate is low the learning is slow and can stuck in a local minima
3. Bad performance for very high dimensional and complex data patterns

Currently we are working to overcome the limitations above by using a novel algorthim we call it **Predictive Hebbian Unified Neurons** or PHUN which shows promising results. For more info about PHUN visit us at: https://www.brainxyz.com/ 
