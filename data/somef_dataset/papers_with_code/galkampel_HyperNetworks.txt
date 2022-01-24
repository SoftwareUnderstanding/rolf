# HyperNetworks (PyTorch)

In this repository we implement a 2-layer GNNs with hypernetworks model on Pubmed citation network with a semi-supervised learning settings (as in https://arxiv.org/pdf/1710.10903.pdf) and NMP-edge with hypernetowrks extension on QM9 (as depicted in https://arxiv.org/pdf/2002.00240.pdf).


A two-layer GNNs with hypernetworks model flow:
![a two-layer GNNs with hypernetworks flow](../master/Figure/2LayerGNNHypernetworks.PNG)

main files:
* train.py - train a 2-layer GNNs, where the models are either GCN (https://arxiv.org/pdf/1609.02907.pdf) or GAT (https://arxiv.org/pdf/1710.10903.pdf)
* train_nmp.py- train nmp-edge with hypernetowrks extension on QM9
* test_nmp.py - test a pretrained nmp-edge with hypernetowrks extension model on QM9

main directories:
* model- a folder where all the models are implemented (GCN, GAT, 2-layer GNNs with hypernetworks (GATGCN) and nmp-edge (nmp-edge with hypernetwork extension))
* input- input files which consist of the parameters to be fed into train.py


*The implementations require PyTorch Geometric (PyG) library


Requirements for Pytorch Geomtric installation:
* At least PyTorch 1.4.0
* At least cuda 10.0.130


PyTorch Geometry Documentation:
https://pytorch-geometric.readthedocs.io/en/latest/
