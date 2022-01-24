# RNA sub-cellular localization with fast-training Quasi-Recurrent Neural Networks

This repo contains the code and the report of a final project for the course **02456 Deep Learning** held at the Technical University of Denmark, Lyngby, by professor Ole Winther.  The paper and poster produced during the project can also be found in the repo.

### Authors
- Alessandro Montemurro, DTU Compute
- Léa Riera, DTU Compute
- Niels MK, DTU Bioengineering  

### Supervisors
- Alexander Rosenberg Johansen
- José Armenteros
 
 
### Description
**Quasi-Recurrent Neural Networks (QRNNs)** are used for the RNA sub-cellular localization (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5210605/).  
QRNNs embrace the benefits of both convolutional and recurrent neural networks alike. QRNNs beat other networks such as LSTM in both accuracy and speed.  
The followind figure depicts the difference between an LSTM layer and a QRNN layer. The first is totally sequential; the second can be parallelized with convolutions and the sequential part in much lighter.
![layer](https://github.com/AllenMont/qrnn-rna-localization/blob/master/img/layer.PNG)

A description of QRNNs can be found in https://arxiv.org/abs/1611.01576. The PyTorch implementation follows https://github.com/salesforce/pytorch-qrnn.

### Running
1. A CUDA-enabled GPU is required for running the network.
2. Install Python package requirements from **requirements.txt**. 
3. Run notebook **QRNN-train.ipynb**. Note: a pretrained model is used by default. 
