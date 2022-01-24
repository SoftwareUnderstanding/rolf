

[image_1]: ./images/acc_loss_from_scratch_adam.png
[image_2]: ./images/tsne_projection.png
[image_3]: ./images/cct_embedding2.png
[image_4]: ./images/roc_curve_Infected.png
[image_5]: ./images/roc_curve_Exposed.png
[image_6]: ./images/roc_curve_Susceptible.png
[image_7]: ./images/confusion_matrix_Susceptible.png

# Motivation

The goal of this repo is to demonstrate the use of the stellargraph implementation of the graphsage algorithm for graph node inference to develop a graph embedding from a learned covid19 contact network to predict the risk classification of novel contacts based on prior knowledge of contacts and a vulnerability measure. The resulting low-dimensional embedding of the grapsage output layer stack can reveal useful contact proximity information not readily available based solely on contact lists or vulnerability data alone. 

# Introduction

Deep-trace is a graphsage-based machine-learning pipeline for contact tracing. Conventional methods can only exploit knowledge of an individual person's contacts. Taken over the set of all individuals, this contact set is essentially a graph with nodes representing people and edges connecting contact between people. The proposed method allows us to utilize information stored in the graph contacts as well as node features to develop a method to classify individuals in the contact set as either susceptible, exposed or infected. In this particular case we use the covid vulnerability index to assign a feature vector to each node. We are then able to learn the contact network based not only on the graph node and edgelist specification, but also the vulnerability feature mapping. Thus we create a three-dimensional node embedding for new contacts that shows an assessment of their likelihood of being in one of three exposure categories – Infected, Exposed or Susceptible. This low dimensional embedding allows contact tracing personnel to prioritize which individuals they should contact and test in situations where a pandemic is evolving too quickly under limited personnel and test resources to correspond with everyone in the contact set. Thu one can quickly identify and prioritize which persons to contact and isolate.
<br />
#

Figure 1 below shows a TSNE projection of the data onto three dimensions for a simulated case study of 27 infected, 519 susceptible, and 419 exposed individuals:

![alt text][image_2]

This is a 2-D projection of the same TSNE embedding:

![alt text][image_3]

## Training 

Accuracy and loss plots for the training dataset during the initial from scratch training:

![alt text][image_1]

### Dataset:

The dataset consists of fictional contacts using the Cora dataset link data and the Covid19 vulnerability example feature data found here: https://github.com/closedloop-ai/cv19index.

## Performance Analysis:

The following ROC curve shows the performance on test data for the infected, exposed and susceptible test classes, respectively:

![alt text][image_4]

![alt text][image_5]

![alt text][image_6]

Confusion matrix for the susceptible class:
![alt text][image_7]

## Critical Dependencies:

* Stellargraph
* NetworkX
* Sklearn
* Python3
* Tensorflow  >= 2.0
* Keras > 2.3
* Pandas

Using Anaconda:
conda env create -f deep-trace.yml

Note: the requirements.txt contains many extraneous packages used in other projects, so you won't need all of them.

## References

***

stellargraph: https://pypi.org/project/stellargraph/<br />
graphsage paper: https://arxiv.org/pdf/1706.02216.pdf<br />
graph node embeddings: https://github.com/stellargraph/stellargraph/blob/develop/demos/node-classification/graphsage-node-classification.ipynb<br />
compartmental modeling: https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology <br />

