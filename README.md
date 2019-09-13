# evolve
A simple evolutionary algorithm (Simultaneous update of all the weights) implemented both in python and C++.
The algorithm is very simple conceptually. it competes between two networks: father (f) & child (c) network and simultaneously mutates all the weights of the child network then chooses the better network to become the father of the next generation. For low dimensional data, this algorithm can be as efficient as naive gradient decent and that is because it doesn't require the backward pass (back-propagtion), instead it requires 2 forward passes (which are less intensive computationally than the backward pass). Also, if the loss function is not convex or the local minima hides in a narrow and elongated ridge, then this algorithm can be more efficient than the naive gradient decent which produces a zigzagged path towards the local minima.

link to google colab (python):

https://colab.research.google.com/drive/18Q9W9WcjnWt8sVcWRIORb1KWrRry6-5L
