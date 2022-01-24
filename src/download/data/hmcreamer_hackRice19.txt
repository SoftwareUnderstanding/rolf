# DynaML - A research toolkit for understanding how misinformation flows through a social network using a dynamic system simulation and graph neural networks

## Inspiration
Understanding the spread of information and particularly disinformation is vital to today's world. Data on the spread of competing narratives by bot agents is difficult to gather at scale over extended periods. We provide a simulation tool kit that enables researchers to understand this problem by model the interaction of bot agents competing to spread different narratives to a broader community.

## Simulating Social Networks as Dynamic Systems
We created a data science tooling system that enables researchers to understand social networks better. In the initial step of our simulation, we create randomized graphs that are similar in edge distributions to real-world social networks. Then for each of the nodes in the graph, we assign an agent type (non-bot neutral, bot promoting idea A, bot promoting idea B). We initialize the spread of information through the graph by beginning with the bots as the only ones trying to spread their respective ideas. Then at each time step we use, the edges of the graph, the identity of the agent, and the current held the belief of each agent to determine a percent chance that information is spread to a new agent. We keep track of the history of all of the information in the network to create our simulated data set. To visualize this complex system of interactions we created an animated assistant that shows off all of the information of the graph. Each node has an identity denoted by its inner darker color, the current idea that the node is spreading is shown by the outer lighter color, and the active connections in the graph are colored edges.

### Installing via PyPI

You can pip install our library via the following pip command:

`pip install dynaml-lib`

Then import the library in your python project by:

`import dynaml_lib`

or access the experiment class with:

`from dynaml_lib.experiment import Experiment`

## Analyzing Social Networks via Graph Neural Networks (GNNs) - An example of research using our network simulation toolkit

With DynaML, we take the stance that the social relationships between users are a necessary component to understanding the spread of information in dynamic social networks. To model these relationships, we build out detailed graph representations as described above. But once we have these simulations, the question becomes: *can we make meaningful inferences on this data*?

### Our learning problem
As an example of what researchers could do with our network simulator toolkit, we took on the problem of predicting if a node is a bot based on the information propagated through the node's local community.

![learning problem diagram](https://i.imgur.com/8Yfam57.png)

**Input:** A social network graph simulated using our toolkit


**Output:** A label of bot (1) or not bot (0) on each node (user in the network)

**The Model:** We based our model on seminal literature in the space of GNNs. Specifically, we focused on the GraphSAGE model (read more about it below or check out our training code on GitHub [here](https://github.com/hmcreamer/hackRice19/tree/master/graph_nn_model)).

**Initial Results:**
We were able to develop the model such that it could train on the simulated data from our toolkit. Find the actual loss in one of our experiments below:

![Loss](https://i.imgur.com/qLJ4xG1.png)

### What are Graph Neural Networks?

You may have heard of fully-connected NNs, Convolutional NNs, or even Recurrent NNs but none of these models are designed to learn on graphical data. Graphical data is everywhere from social networks to protein folding. Essentially, **GNNs are designed to take advantage of graphical structure to make better predictions**.

### GraphSage

This Graph Neural Network model explores a node's neighbors across connections and aggregates this 'communal' information to understand the content and relationships contained in the graph. See the graphic below:

![https://arxiv.org/pdf/1706.02216.pdf](https://i.imgur.com/tlLIHZa.png)

**Check out the paper:**
Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." Advances in Neural Information Processing Systems. 2017.

### PyTorch Geometric

While PyTorch on its own is a great deep learning framework, it has not yet committed to handling graphical data. To get around this, we used the [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) library, which provides additional functionality to PyTorch to process graph data. Learning to use this framework to load data and train a GNN was very rewarding and insightful.

**Read the paper on it:**
Fey, Matthias, and Jan Eric Lenssen. "Fast graph representation learning with PyTorch Geometric." arXiv preprint arXiv:1903.02428 (2019).


## Challenges we ran into
One challenge that we ran into was visualizing these graphs at scale. We created several networks with thousands of nodes in them but were unable to effectively visualize these graphs due to a large number of nodes and edges. We even trained our graph neural network on these large graphs but were unable to visualize the results. 
Another issue we had was removing a huge file from Git after we accidentally committed it (oops). 

## Accomplishments that we're proud of
-Successfully created a dynamic simulation of how competing ideas spread in a social network filled with non-bots and bots
-Successfully created a dynamic visualization of the spread of competing for information in a network. This involved showing the original status of the different nodes (i.e., non-bot, bot promoting point A, bot promoting point B), the current state of the node, and the destinations of where each node was trying to spread its information
-Visualized how non-bots start spreading misinformation in a network once influenced by bots
- Used PyTorch to implement a graph neural network to predict the likelihood of a user in a social network is a bot
-Created a Python library for our code that will allow researchers to easily use our tool for conducting research on the spread of misinformation in social networks so they can help stop this problem

## What we learned
-How to dynamically simulate interactions between nodes in a graph/network
-How to train and test a graph neural network used to accurately predict attributes of nodes in the graph
-How to dynamically visualize/animate a graph using Javascript
-How to develop a functional web application using Flask

## Future Developments for of DynaML
-We want to increase the customizability of the networks so that our simulation tool kit can be used by researchers in multiple areas of interest (diseases spreading, computer network attacks, infrastructure grids)
-We want to investigate combining time series analysis models such as state-space with state of the art graph neural networks for predicting how the information will spread multiple time steps before
-We want to show that methods developed using our simulations are generalizable to real-world data sets
-We want to be able to visualize dynamic graphs at the scale of hundreds of thousands of nodes and edges. 
