# gcn-collisions
This Graph Convolutional Network learns to simulate collision responses for particles. Speedups can be seen for large numbers of particles. The network can also learn to account for symmetry-breaking from a reference algorithm.

## Notice

If you do decide to use the codebase, I would appreciate a mention of my name in any releases. Feel free to also send me a message, I'd love to see what you are developing.

### Credits
I haven't used anyone else's code, but this idea stemmed from Deepmind's "Learning to Simulate Complex Physics with Graph Networks (ICML 2020)". I feel like some credit therefore is due. Do take a look at their links, the project is very impressive and much larger in scope than this one.

#### Deepmind's "Learning to Simulate Complex Physics with Graph Networks (ICML 2020)"
ICML poster: icml.cc/virtual/2020/poster/6849

Video site: sites.google.com/view/learning-to-simulate

ArXiv: arxiv.org/abs/2002.09405

## Used methods

### Physics-based simulation
To gather data, I made a physics simulation of colliding particles in a box. The physics is correct for pairwise collisions. However, for multi-particle collisions, the serial nature of the algorithm actually allows for inconsistencies. In a three-particle collision, for example, particles 1 and 2 can first collide, after which they collide with particle 3. Another possibility is that particle 1 and 3 collide, after which they collide with particle 2. These cases give different dynamics. Simulating the true process is more demanding with correct physics.

### Machine Learning simulation
To see how well Machine Learning algorithms could handle this situation, as well as how it scales with number of particles, I made a Graph Convolutional Network (GCN) and trained it with the physics simulation. The GCN takes the positions and velocities of the particles, along with an adjacency matrix where the indices correspond to which particles collide with one another. 

### Collision indices
To find out which particles collide efficiently, I used Scipy's cKDTree, which is a Q-Tree implemented in C. 

## Comparison of methods

### General dynamics 

#### Physics-based simulation
There isn't much to complain about in this general setting. Since the physics simulation is based on conservation of momentum and energy, this perfectly describes the collision responses between two particles. However, as we'll see below, collisions between three particles are not correct. 



https://user-images.githubusercontent.com/54723095/133246681-f77332cf-7166-46eb-9c0c-57c9b9514472.mp4



#### GCN 
There seems to have been a slight loss in energy from the physics based simulation to the GCN (although this can be compensated for by restoring the enrgy to the previous time step), but overall, the collision responses look very realistic. 



https://user-images.githubusercontent.com/54723095/137793923-dfb24f57-c8c7-49fd-ac99-563ad84d9921.mp4






### Symmetry

#### Physics-based simulation
An obvious break of symmetry is seen in the sequential physics simulation. Clearly, the physics are not quite correct for simultaneous collisions between multiple particles. To my understanding, finding the resulting velocities in a three-way collision between particles is not analytically solvable. In reality, the probability of a simultaneous collision between particles is vanishingly small, but in a programming setting, having discrete time steps, this is not the case. One way in which we could treat the problem more consistently would be to perform collisions in increasing order of pairwise distances. The particles with the smallest distances between each other then collide first. Nevertheless, observing the below video, the resulting velocities do not look right.


https://user-images.githubusercontent.com/54723095/132208241-4898255b-7ca5-44a7-8f2b-eaf2519fe5bc.mp4



#### GCN
The GCN learns to account for the inconsistencies. This is likely due to that the GCN has seen many multi-particle collisions, each with different dynamics from the physics simulation. The average of the possible collision responces in the above situation would, however, be symmetric. Meanwhile, the loss function is mean squared error (MSE). Because of the quadratic scaling of MSE, minimising the loss function would correspond to finding an average between the observed inconsistencies.


https://user-images.githubusercontent.com/54723095/132208225-e0c65715-84b7-4847-b2e9-abd339353251.mp4


