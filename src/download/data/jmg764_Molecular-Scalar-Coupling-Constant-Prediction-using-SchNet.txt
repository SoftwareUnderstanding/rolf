# Molecular Scalar Coupling Constant Prediction using SchNet
This final project for the CS-GY-9223 Deep Learning course at NYU Tandon implements SchNet, based on the [paper by Schütt et al.](https://arxiv.org/abs/1706.08566), for prediction of molecular scalar coupling constants.


## 1. Introduction
The drug discovery process is one of the most challenging and expensive endeavors in biomedicine. While there are about <img src="https://render.githubusercontent.com/render/math?math=10^56"> atoms in the solar system, there are about <img src="https://render.githubusercontent.com/render/math?math=10^60"> chemical compounds with drug-like features that can be made. Since it is unfeasible for chemists to synthesize and evaluate every molecule, they’ve grown to rely on virtual screening to narrow down promising candidates. However, the challenge of searching this almost infinite space of potential molecules is the perfect substrate for deep learning techniques to improve the drug discovery process even further. While the growing number of large datasets for molecules has already enabled the creation of several useful models, the application of deep learning to drug discovery is still in its infancy. Some useful predictions that could expedite drug discovery include toxicity, ability to bind with a given protein, and quantum properties.

Researchers commonly use Nuclear Magnetic Resonance (NMR) to gain insight into a molecule’s structure and dynamics. NMR’s functionality largely depends on its ability to accurately predict scalar couplings which are the strength of magnetic interactions between pairs of atoms in a given molecule. It is possible to compute scalar couplings on an inputted 3D molecular structure using advanced quantum mechanical simulation methods such as Density Functional Theory (DFT) which approximate Schrödinger’s equation. However, these methods are limited by their high computational cost, and are therefore reserved for use on small systems, or other, less approximate, methods are adopted instead. My goal for this project was to develop a fast, reliable, and cheaper method to perform this task through the use of a graph convolutional neural network (GCN). In particular, I focused on implementing and optimizing SchNet: a novel GCN that has been shown to achieve state-of-the-art performance on quantum chemical property benchmarks. As a byproduct, I hoped to learn more about GCNs and how they could be used for chemical applications.

## 2. Literature Survey
### a. GCN motivation and basics

Many of the deep learning models designed to aid in drug discovery show improvement over traditional machine learning methods, but are limited due to two main reasons: first, they rely on hand-crafted features which prevents structural information to be learned directly from raw inputs, and, second, the existing architectures are not conducive for use on structured data such as molecules. Extraction of relevant features from images have already proven highly successful using convolutional neural networks (CNNs). Molecules can be represented as fully connected graphs in which atoms and bonds can be represented as nodes and edges, respectively. Graphs are irregularly shaped thereby making CNNs, which rely on convolution on regular grid-like structures, unsuitable for feature extraction [1]. 

Efforts have been made to generalize the convolution operation for graphs, resulting in the development of graph convolutional neural networks (GCNs). As Kipf and Welling describe in their seminal paper [2], the idea behind graph convolutional neural networks (GCNs), as shown in Fig. 1, is to perform convolutions on a graph by aggregating (through sum, average, etc) each node’s neighborhood feature vectors. This new aggregated vector is then passed through a neural network layer, and the output is the new vector representation of the node. Additional neural network layers repeat this same process, except the input is the updated vectors from the first layer. 

<p align="center">
  <img src="images/Figure 1.png"  alt="drawing" width="600"/>
</p>

### b. Quantum mechanical property prediction

In 2017, Gilmer et al. [3] released a paper focusing on the specific use of neural networks for predicting quantum properties of molecules. They noted that the symmetries of atomic systems require graph neural networks that are invariant to graph isomorphism, and therefore reformulated existing models that fall into this category, including Kipf and Welling’s GCN, into a common framework called Message Passing Neural Networks (MPNNs). The “message passing” refers to the aggregation of neighborhood vector features described earlier. The MPNN that Gilmer et al. developed, called enn-s2s, managed to achieve state-of-the-art performance on an important molecular property benchmark using QM9: a dataset consisting of 130k molecules with 13 properties for each molecule as approximated by DFT. The neighborhood messages generated used both bond types and interatomic distances followed by a set2set model from Vinyals et al. [4]. 

Later on, Schutt et al. pointed out that enn-s2s was limited by the fact that atomic positions are discretized, and therefore the filter learned was also discrete which rendered it incapable of capturing the gradual positional changes of atoms [5]. In order to remedy this, Schutt et al. proposed a different method of graph convolution with continuous filters that mapped an atomic position to a corresponding filter value. This is advantageous in that it doesn't require atomic position data to lie on a grid, thereby resulting in smooth, rather than discrete energy predictions.

<p align="center">
  <img src="images/Figure 2.png"  alt="drawing" width="500"/>
</p>

SchNet demonstrated superior performance over enn-s2s in predicting molecular energies and atomic forces on three different datasets. Fig. 3 provides an overview of the SchNet architecture. Molecules input into the model can be uniquely represented by a certain set of nuclear charges <img src="https://render.githubusercontent.com/render/math?math=Z = (Z_1, ..., Z_n)"> and atomic positions <img src="https://render.githubusercontent.com/render/math?math=R = ({\rm r}_1, ... , {\rm r}_n)"> where n is the number of atoms. At each layer, the atoms in a given molecule are represented as a tuple of features: <img src="https://render.githubusercontent.com/render/math?math=X^l = ({\rm x}^l_1, ..., {\rm x}_n^l)"> with <img src="https://render.githubusercontent.com/render/math?math={\rm x}_i^l \in \mathbb R^F"> where <img src="https://render.githubusercontent.com/render/math?math=l"> and <img src="https://render.githubusercontent.com/render/math?math=F"> are the number of layers, and feature maps, respectively. This representation is analogous to pixels in an image. In the embedding layer, the representation of each atom  is initialized at random using an embedding dependent on the atom type <img src="https://render.githubusercontent.com/render/math?math=Z_i"> which is optimized during training: <img src="https://render.githubusercontent.com/render/math?math=x^0_i = {\rm a}_{Z_i}"> where <img src="https://render.githubusercontent.com/render/math?math=a_Z"> is the atom type embedding. 

Atom-wise layers, a recurring building block in this architecture, are dense layers that are applied to each representation <img src="https://render.githubusercontent.com/render/math?math={\rm x}_i^{l}"> of atom. These layers are responsible for the recombination of feature maps with shared weights across all atoms which allows the architecture to be scaled with respect to the size of the molecule. 

Interactions between atoms are modeled by three interaction blocks: as shown above, the sequence of atom-wise, interatomic continuous-filter convolution (cfconv), and two more atom-wise layers separated by a softplus non-linearity produces <img src="https://render.githubusercontent.com/render/math?math={\rm v}_i^l">. The cfconv layer uses a radial basis function that acts as a continuous filter generator. Additionally, the residual connection between <img src="https://render.githubusercontent.com/render/math?math=({\rm x}^l_1, ..., {\rm x}_n^l)"> and <img src="https://render.githubusercontent.com/render/math?math=({\rm v}^l_1, ..., {\rm v}_n^l)"> allows for the incorporation of interactions between atoms and previously computed feature maps. 

<p align="center">
  <img src="images/Figure 3.png"  alt="drawing" width="600"/>
</p>

## 3. Chainer Chemistry Implementation
### a. Dataset

I used the CHAMPS Scalar Coupling dataset which was provided for a Kaggle competition with a similar objective [6], and consists of the following: 
<ul type="disc">
  <li> train.csv — the training set which contains four columns: (1) the name of the molecule where the coupling constant originates, (2) and (3) the atom indices of the atom-pair which create the coupling, (4) the scalar coupling type, (5) the scalar coupling constant that we want to predict.</li> 
  <li> scalar_coupling_contributions.csv — the scalar coupling constants in train.csv are a sum of four terms: Fermi contact, spin-dipolar, paramagnetic spin-orbit, and diamagnetic spin-orbit contributions which are contained in this file. It is organized into the following columns: (1) molecule name; (2) and (3) the atom indices of each atom-pair; (4) the type of coupling; and (5), (6), (7), and (8) are the four aforementioned terms.</li>
  <li> structures.csv — contains the x, y, and z cartesian coordinates for each atom in each molecule. It is organized into the following columns: (1) molecule name; (2) atom index; (3) atom name; and (4), (5), and (6) are the x, y, and z cartesian coordinates, respectively.</li>
</ul>

### b. Preprocessing

I aimed to ultimately train a SchNet model that predicts the scalar coupling contributions instead of the scalar coupling constant itself because, in general, this sort of multi-task learning helps combat overfitting. I therefore merged train.csv and scalar_coupling_contributions.csv into one Pandas DataFrame, and performed a 80-10-10 (68,009-8501-8502 molecules) split for train, validation, and test sets. 

structures.csv contains cartesian coordinates, so additional preprocessing was required in order to create graph representations of each molecule. In order to do so, I created a Graph class whose objects store the distances between each atom and an adjacency matrix. The fully processed train, validation, and test datasets for input into SchNet were dictionaries consisting of Graph objects for each molecule and associated scalar coupling contributions for each atom pair within that molecule. As an example of how each molecule is represented graphically, Fig. 4 displays the first molecule in the dataset, CH4 (methane).

<p align="center">
  <img src="images/Figure 4.png"  alt="drawing" width="325"/>
</p>

### c. SchNet Model

I constructed, trained, evaluated, and optimized the SchNet model using the Chainer Chemistry library which uses the same architecture described by Schutt et al. In addition, however, I added batch normalization using Chainer Chemistry’s ```GraphBatchNormalization```. 

The loss function used is log mean absolute error (Log MAE): <img src="https://render.githubusercontent.com/render/math?math={\rm loss} = \frac{1}{T} \sum_{t=1}^{T} {\rm log} \Big (\frac{1}{n_t} \sum_{i=1}^{n_t} |y_i - \hat y_i| \Big )"> where <img src="https://render.githubusercontent.com/render/math?math=T"> is the number of scalar coupling types, <img src="https://render.githubusercontent.com/render/math?math=n_t"> is the number of observations of type <img src="https://render.githubusercontent.com/render/math?math=t">, <img src="https://render.githubusercontent.com/render/math?math=y_i"> is the actual scalar coupling constant for the observation, and <img src="https://render.githubusercontent.com/render/math?math=\hat y_i"> is the predicted scalar coupling constant for the observation. It is calculated for each scalar coupling type, and then averaged across tips, so that a 1% decrease in error for one type provides the same improvement in score as a decrease for another type.

I tested out a variety of hyperparameter configurations in order to optimize the model; each of them used Adam optimization with a batch size of 4 over the course of 25 epochs. I was particularly focused on discovering the optimal radial basis function hyperparameters within the cfconv layer. The radial basis function is expressed as <img src="https://render.githubusercontent.com/render/math?math=e_k({\bf r}_i - {\bf r}_j) = {\rm exp}(-\gamma \|d_{ij} - \mu_k \|^2)"> located at centers 0Å ≤ µ_k ≤ 30Å every 0.1Å with γ = 10Å, and 300 is the interatomic distance used as input for the filter network. This translates to hyperparameter values ```num_rbf```=300, ```radius_resolution```=0.1, and ```gamma```=10.0 in Chainer Chemistry implementation of cfconv. In general, these values are chosen such that all distances occurring in the datasets are covered by the cfconv filters. Choosing fewer centers corresponds to reducing the resolution of the filter, while restricting the range of the centers corresponds to the filter size in a usual convolutional layer.

Using these default cfconv hyperparameter values, I ran SchNet using Adam learning rates of <img src="https://render.githubusercontent.com/render/math?math=1\times 10^{-3}">, <img src="https://render.githubusercontent.com/render/math?math=5\times 10^{-3}">, and <img src="https://render.githubusercontent.com/render/math?math=1\times 10^{-2}">. Each 25-epoch training took approximately 4.5-5 hours. As shown in the table below, the model performed best (achieved lowest Log MAE) with a learning rate of <img src="https://render.githubusercontent.com/render/math?math=5\times 10^{-3}">.

 | **Learning Rate** | **Log MAE** |                 
 | :---: | :---: |  
 | 1x10^-3 | -1.22 |
 | 5x10^-3 | -1.32 |
 | 1x10^-2 | -1.17 |            


Using this optimal learning rate, I then ran the model using different values of radius resolution, and found that the lowest validation Log MAE was achieved with a radius resolution of 0.10:

 | **Radius Resolution** | **Log MAE** |                 
 | :---: | :---: |  
 | 0.05 | -1.21 |
 | 0.075 | -1.25 |
 | 0.10 | -1.32 |   
 | 0.125 | -1.18 |   
 | 0.15 | -1.27 |   

Below is a sample of the 1,400,457 predicted scalar coupling constants predicted on the test dataset using these optimal learning rate and radius resolution values:

 | **Atom ID** | **Predicted Scalar Coupling Constant** |  **Actual Scalar Coupling Constant** |                
 | :---: | :---: | :---: | 
 | 236 | 79.78 | 79.09 |
 | 237 | 1.89 | 1.45 |
 | 238 | 0.94 | 0.93 |
 | 239 | -12.34 | -12.09 |
 | 240 | -11.36 | -11.28 |
 | 241 | 1.69 | 1.43 |
 | 242 | 79.86 | 79.64 |
 | 243 | -2.80 | -3.07 |
 | 243 | 8.45 | 8.26 |


## 4. Conclusion

Thus, using data provided from the CHAMPS Scalar Coupling dataset, I was able to successfully train a SchNet model in order to predict scalar coupling constants for each pair of atoms in the molecules provided. In order to optimize the model, I manually tested out a variety of learning rates and radius resolution values, and achieved optimal performance using a learning rate of 5x10^-3 and radius resolution of 0.10. The 4.5-5 hour training time produced results with low Log MAE which is a drastic improvement over the days or weeks it would take produce similar results using quantum mechanical simulation methods such as Density Functional Theory (DFT). In the future, I would like to perform an even more thorough hyperparameter grid search, and run SchNet using all combinations of different values for ```num_rbf```, ```radius_resolution```, and ```gamma```. 

## 5. References

<ol type="disc">
<li>Sun, M., Zhao, S., Gilvary, C., Elemento, O., Zhou, J., & Wang, F. (2019). Graph convolutional networks for computational drug development and discovery. Briefings in Bioinformatics, 21(3), 919-935. doi:10.1093/bib/bbz042</li>
<li>Kipf, Thomas N. and Welling, Max. Semi-Supervised Classification with Graph Convolutional Networks. International Conference on Learning Representations (ICLR), arXiv:609.02907v4, 2017.</li>
<li>Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212, 2017.</li>
<li>Vinyals, Oriol, Bengio, Samy, and Kudlur, Manjunath. Order matters: Sequence to sequence for sets. arXiv preprint arXiv:1511.06391, 2015.</li> 
<li>Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Rober Müller. Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems (NIPS) 30, pages 992–1002. Curran Associates, Inc., arXiv: 1706.08566v5, 2017.</li>
<li>Predicting Molecular Properties. https://www.kaggle.com/c/champs-scalar-coupling/data</li>
</ol>




