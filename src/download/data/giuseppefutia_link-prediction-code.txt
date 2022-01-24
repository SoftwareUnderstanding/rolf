# Link Prediction Task
This repository provides an experiment on link prediction tasks using the [ogbl-ddi dataset](https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi). The dataset is a homogeneous, unweighted, undirected graph representing the drug-drug interaction network. Using a feature-less approach (one-hot encoding representation of the nodes), I will try to test the capability of a Graph Neural Network (GNNs) model to reconstruct the edges in the original graph.

## Dataset
The ogbl-ddi graph includes:
* 4267 nodes
* 2135822 edges

The edges are splitted as follows:
* training edges: 1067911
* valid edges: 133489
* test edges: 133489

The good practice is to select the model that achieves the best performance on the validation dataset after the training process. However, considering the amount of available time for this experiment, I will skip this step, and I will test the link prediction mechanism directly on the test edges, using the model resulting from the last epoch of the training process.

## Evaluation Metric
The evaluation metric for this experiment is the Mean Reciprocal Rank (MRR). In a nutshell, it is the count of how many correct links (positive examples) are ranked in the top-n positions against a bunch of synthetic negative examples. In this specific experiment, for each drug interaction, a set of approximately 100.000 randomly sampled negative examples are created. Then, I compute the count of the ratio of positive edges ranked at the K-place or above (Hits@K).

To clarify this aspect, imagine the test set includes two ground truth positive examples:
* source: node1, target: node2
* source: node4, target: node5

I create five negative examples for each right edge and then compute the trained model's link prediction score. 

For the first edge, the results are the following:

| s     | t     | score | rank |   |
|-------|-------|-------|------|---|
| node1 | node3 | 0.790 | 1    |   |
| node1 | node2 | 0.789 | 2    | * |
| node4 | node1 | 0.695 | 3    |   |
| node2 | node5 | 0.456 | 4    |   |
| node4 | node6 | 0.234 | 5    |   |

For the first edge, the results are the following:

| s     | t     | score | rank |   |
|-------|-------|-------|------|---|
| node4 | node5 | 0.779 | 1    | * |
| node1 | node3 | 0.743 | 2    |   |
| node4 | node1 | 0.695 | 3    |   |
| node2 | node5 | 0.456 | 4    |   |
| node4 | node6 | 0.234 | 5    |   |

Then, if we want to compute Hits@1 and Hits@3 we count how many positives occur in the top-1 or top-3 positions and divide by the number of edges in the test set (which in this example includes two edges):

Hits@3= 2/2 = 1.0
Hits@1= 1/2 = 0.5

## Installation
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Tested with Python 3.8.5.

# Approach and Experiment Details
The experiment has been performed using hardware with the following features:
* RAM size: 251 GiB
* GPU: NVIDIA GeForce RTX 3090 (shared with other processes)

I will go to test two different models GCN (https://arxiv.org/abs/1609.02907) and GAT(https://arxiv.org/abs/1710.10903). But, first, I will try to make some preliminary assumptions to avoid performing an entire training process for both the models and save time.

## Preliminary Tests
Before running a real experiment, I need to make some tests on the training process to ensure that the process concludes with unexpected interruptions. It can be helpful to avoid discovering bugs only at the end of the training step.

To start a speedy training process, you can run the following command:

```
python embeddings.py --n-epochs 1 --n-samples 10 --hidden-dim 8
```

### Hypothesis on Training Time
The further step is to estimate the total training time. For this reason, I run 5 epochs using the original configuration. I test 5 epochs because, considering the dimension of the training graph, it can not be directly used for the training process. For this reason, I apply a sampling process, which selects a fixed number of neighbors for all the nodes (https://docs.dgl.ai/en/0.6.x/generated/dgl.sampling.sample_neighbors.html). Then, I want to test that each epoch has the same duration. Considering a `--n-sample = 500` hyperparameter, the number of sampled edges for each epoch is around 30.000 over 1067911.

```
python embeddings.py --n-epochs 5
```

Training 5 epochs took around 6 minutes for both GCN and GAT. This time was predictable because the two models have a close number of parameters:
* GCN: 68609
* GAT: 68849

The additional parameters to GAT are related to the attention coefficients.

Observing the result, the training of 5 epochs took around 6 minutes. As a consequence, in one hour, we can train around 50 epochs.

### Testing the Model with Low Number of parameters and samples
I decided to perform a 1-hour training for both the models using a very low number of parameters and edge samples. The results achieved by both the models are in terms of MRR are:

#### GCN
Test:
* Hits@10: 0.0007% 
* Hits@50: 0.0060% 
* Hits@100: 0.0217% 

Train:
* Hits@10: 0.0000% 
* Hits@50: 0.0062% 
* Hits@100: 0.0179% 

#### GAT
Test:
* Hits@10: 0.0000% 
* Hits@50: 0.0000% 
* Hits@100: 0.0000% 

Train:
* Hits@10: 0.0002% 
* Hits@50: 0.0004% 
* Hits@100: 0.0004%

The GCN model seems the most promising one. The more likely reason is that the edge sampling process does not allow attention coefficients to be learned correctly. Indeed, the attention coefficients try to grasp the relevance of each neighbor for the central node. Unfortunately, due to the sampling process, the local graph structure around each node changes for each training step.

## First Experiment and Suggestion to Improve the Performance
For this experiment, I decided to increase the value of some hyperparameters and check the impact in terms of MRR values. I applied the following changes:
* 2 GCN layers
* 1000 samples

Adding another GCN layer allows aggregating the features of the 2-hop neighbors. Consider this new sampling value. The number of sampled edges for each epoch is around 70.000 over 1067911. The training duration has been of 3.7 hours. The resulting value in terms of MRRs are:

Test:
* Hits@10: 0.0210%
* Hits@50: 0.0479%
* Hits@100: 0.0674%

Train:
* Hits@10: 0.0254%
* Hits@50: 0.0392%
* Hits@100: 0.0441%

The performances have increased significantly, but there is still considerable room for improvement. In this case, the goal was to develop an approach in a few hours, detect the right direction for selecting the correct model, and set the hyperparameters in about half a day. Moreover, I wanted to use almost real data (ogb-ddi includes more than 2 million edges), avoid toy data such as Zachary's karate club (78 edges in its base version).


To improve the performance we can proceed in the following directions:
* Extend the number of samples: the best situation would be not using samples, in order to process the entire graph dataset for each epoch. Alternatively, increasing the number of epoch is a good trade off.
* Increase th number of parameters: compared to in-production models, which can include millions of parameters, the current model does not overcome the number of 70.000.
