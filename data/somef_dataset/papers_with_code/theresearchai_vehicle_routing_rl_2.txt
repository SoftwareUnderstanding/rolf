# Vehicle routing using reinforcement learning*

## Introduction
* The Vehicle Routing Problem is a combinatorial optimization problem which asks "What is the optimal set of routes for a fleet of vehicles to traverse in order to deliver to a given set of customers?“
* Capacitated Vehicle Routing Problem (CVRP) is a variant of the Vehicle Routing Problem in which the vehicles have a limited capacity of the goods.
* This repository leverages Deep Reinforcement Learning to solve the CVRP problem. 


## Background Knowledge
* The Vehicle Routing Problem is an NP-hard problem.
* Aim is to find sub-optimal heuristics that run within a reasonable time. 
Finding suboptimal heuristics is a tedious task because of the large number of states and paths. Therefore, Deep Reinforcement Learning (RL) is used to determine these suboptimal heuristics without any human intervention.
* Wouter Kool’s “Attention, Learn to Solve Routing Problems!”  uses an end-to-end approach. It uses an encoder and decoder to train a model to learn to solve the CVRP problem.
* A proximal policy optimization (PPO) algorithm that uses fixed-length trajectory segments. PPO is a family of first-order methods that use a clipping to keep new policies close to old.

####   Wouter Kool’s “Attention, Learn to Solve Routing Problems!”
* The attention paper uses an end-to-end approach. 
* It uses an encoder and decoder to train a model to learn to solve the CVRP problem. 
* We use a system of N nodes such that each node is represented in a 2D coordinate plane to describe the input. The input is fed into the encoder, consisting of a Multi-Headed Attention layer and a Feed-Forward layer.
* Running the input through N sequential layers of encoder we generate two outputs: node embeddings and graph embeddings. The node embeddings are the continuous vector representations of coordinates and the graph embeddings are the aggregated (mean) node embeddings.

![alt text](https://github.com/theresearchai/vehicle_routing_rl_2/blob/main/images/encoder.png)
* The decoder uses the outputs from the encoder along with the outputs from the previous timestamp. The process is sequential in time, which means the decoder produces an output at every timestamp. Combining the inputs in the decoder we generate a context node to represent the decoding context of the problem. Using the context nodes and node embeddings we then work towards generating normalized output probabilities, which then decides the next node in the routing plan. The training of the model is done by minimizing the expected cost of the tour length using a policy gradient method.

![alt text](https://github.com/theresearchai/vehicle_routing_rl_2/blob/main/images/decoder.png)


#### Proximal Policy Optimization (PPO)
* PPO is motivated by the question: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simple to implement.
![alt text](https://github.com/theresearchai/vehicle_routing_rl_2/blob/main/images/ppo.PNG)
* There are two primary variants of PPO: PPO-Penalty and PPO-Clip.
    * PPO-Penalty
    * PPO-Clip
We use the PPO-Clip with the loss as:
![alt text](https://github.com/theresearchai/vehicle_routing_rl_2/blob/main/images/ppo%20loss.png)

#### Weights and Biases
* Weights & Biases is used to track machine learning projects.
* In this project, the product ‘Sweep’ is used for hyperparameter tuning.
* Various combinations of the hyperparameters- learning rates, decay rates and number of epochs are used to tune the model. Different configurations are compared to each other based on average cost/distance.
* The generalizability of the trained model is also checked by running it for different distributions.



## Data
* To generate simulated data, we will leverage this repository.
* Each pickle file contains 10,000 rows and 4 columns.
    * Depot: It is the start and the end point for the vehicle routing problem. For example: An Amazon delivery vehicle starts from the warehouse, delivers all packages and comes back to the warehouse. The data format is a list of length 2. It represents a coordinate in the xy plane.
    * Nodes: The points where the vehicle is required to visit. An effective strategy is required to identify a path to all these points. For example: The delivery address are the points that the Amazon vehicle has to visit. The data format is a list of lists of length 20, 30, or 50. Each inner list is of length 2 that represents a coordinate in the xy plane.
    * Demand: Each node has a demand or a requirement. For example: Every address requires the correct number of packages to be delivered to their address. This represents the demand and is a list of length 20, 30, or 50. Each demand value corresponds to a node.
    * Capacity: It is the maximum capacity of a vehicle. For example: An Amazon truck can carry only x amount of packages in one iteration of traversal to each point. This is a scalar value that represents the capacity of the vehicle.

## Dependencies

* Python>=3.6
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)
* Wandb

### Generating data

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for all problems:
```bash
!python generate_data_validation.py --problem all --name validation --seed 4321 -f
!python generate_data_test.py --problem all --name test --seed 4321 -f
```
### Training

```bash
!python run.py --graph_size 20 --baseline rollout --problem 'cvrp' --n_epoch 10 --epoch_size 1280000 #for PPO reinforce
!python run.py --graph_size 20 --baseline critic_lstm --problem 'cvrp' --n_epoch 10 --epoch_size 1280000 # for PPO LSTM
```

## Conclusions
* Validation results shows comparable performance to State-of-the-Art or reference model.
* Hyperparameter optimization enabled comparable results in just 10 iterations of model training as compared to 100 iterations for State-of-the-Art model.
* The reference model and new implementation work well for different distributions of nodes.
* Few cases result in termination due to routes where the condition of total route demand <= capacity fails.


## Acknowledgements
Thanks to [Wouter Kool’s “Attention, Learn to Solve Routing Problems!”](https://github.com/wouterkool/attention-learn-to-route) for getting us started with the code for the Attention Network.

## Note:
*The repository contains experiments for CVRP as well as other types of problems. Please refer to the tranining commands for CVRP

## References:
   [1] Kool, W. (2018, March 22). Attention, Learn to Solve Routing Problems! Retrieved from https://arxiv.org/abs/1803.08475
   
   [2] Schulman, J. (2017, July 20). Proximal Policy Optimization Algorithms. Retrieved from https://arxiv.org/abs/1707.06347
   
   [3] “TensorBoard:TensorFlow.” TensorFlow, www.tensorflow.org/tensorboard
   
   [4] Weights & Biases – Developer Tools for ML, www.wandb.com/
   
   [5] Proximal policy optimization¶. (n.d.). Retrieved March 04, 2021, from https://spinningup.openai.com/en/latest/algorithms/ppo.html
   
   [6] Sweeps. (n.d.). Retrieved March 14, 2021, from https://docs.wandb.ai/sweeps#common-use-cases  
