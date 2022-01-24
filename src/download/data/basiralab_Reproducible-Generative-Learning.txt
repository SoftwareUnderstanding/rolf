# Reproducible Generative Learning
How to quantify reproducibility of graph neural networks while using generative learning?

Please contact mohammedaminegh@gmail.com for inquiries. Thanks. 

![Reproducible Generative Learning pipeline](main_figure.png)

# Introduction
This work is accepted at the PRIME workshop in MICCAI 2021.

> **Investigating and Quantifying the Reproducibility of Graph Neural Networks in Predictive Medicine**
>
> Mohammed Amine Gharsallaoui, Furkan Tornaci and Islem Rekik
>
> BASIRA Lab, Faculty of Computer and Informatics, Istanbul Technical University, Istanbul, Turkey
>
> **Abstract:** *Graph neural networks (GNNs) have gained an unprecedented attention in many domains including dysconnectivity disorder diagnosis thanks to their high performance in tackling graph classification tasks. Despite the large stream of GNNs developed recently, prior efforts invariably focus on boosting the classification accuracy while ignoring the model reproducibility and interpretability, which are vital in pinning down disorder-specific biomarkers. Although less investigated, the discriminativeness of the original input features -biomarkers, which is reflected by their learnt weights using a GNN gives informative insights about their reliability. Intuitively, the reliability of a given biomarker is emphasized if it belongs to the sets of top discriminative regions of interest (ROIs) using different models. Therefore, we define the first axis in our work as \emph{reproducibility across models}, which evaluates the commonalities between sets of top discriminative biomarkers for a pool of GNNs. This task mainly answers this question: \emph{How likely can two models be congruent in terms of their respective sets of top discriminative biomarkers?} The second axis of research in our work is to investigate \emph{reproducibility in generated connectomic datasets}. This is addressed by answering this question: \emph{how likely would the set of top discriminative biomarkers by a trained model for a ground-truth dataset be consistent with a predicted dataset by generative learning?} In this paper, we propose a reproducibility assessment framework, a method for quantifying the commonalities in the GNN-specific learnt feature maps across models, which can complement explanatory approaches of GNNs and provide new ways to assess predictive medicine via biomarkers reliability. We evaluated our framework using four multiview connectomic datasets of healthy neurologically disordered subjects with five GNN architectures and two different learning mindsets: (a) conventional training on all samples (resourceful) and (b) a few-shot training on random samples (frugal).*


## Code
This code was implemented using Python 3.8 (Anaconda) on Windows 10.

## Installation
### *Anaconda Installattion*
* Go to  https://www.anaconda.com/products/individual
* Download version for your system (We used Python 3.8  on 64bit Windows 10 )
* Install the platform
* Create a conda environment by typing:  ```conda create –n env_reproducibility pip python=3.8 ```

### *Dependency Installattion*
Copy and paste following commands to install all packages (CPU version)
```sh
$ conda activate env_reproducibility
$ conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
$ pip install scikit-learn
$ pip install matplotlib
$ pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html
$ pip install torch-geometric
$ pip install annoy
$ pip install fbpca
```
These instructions are for CPU installation. If you want GPU installation, please visit (optional) PyTorch-Geometric’s web page (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for description on installing GPU version. Code will check the version of dependencies and availability of GPU. If everything is configured correctly, it will utilize GPU automatically.

## Data format
In case you want to use our framework, the input dataset should be a list of numpy arrays. Each numpy array is of size (n_r, n_r, n_v), where n_r and n_v are the number of regions and views, respectively. We provided within this code, the python file ```handle_data/simulate_data.py``` to simulate data. You should specify the number of subjects, the number of views and the number of regions. After simulating the dataset, you can predict the data via generative learning using this file ```topogan/main_topogan.py```. The input data should be vectorized before using the generative learning. To vectorize the data, you can use ```handle_data/vectorize.py```. The output of the generative learning is in a vectorized format. To restore the matrix format of the generated data, you can use this file ```handle_data/collect_generated.py```. After predicting the dataset using generative learning, put both repositories (real and generated) in this path ```reproducibility/data```.

## Run reproducibility framework
After obtaining the real and generated datasets, you can run the GNN models by running this file ```reproducibility/demo.py```. You can open up a terminal at the ```reproducibility``` directory and type in
```sh
$ conda activate env_reproducibility & python demo.py
```
## GNN models
The GNN models included are:
| Model | Paper |
| ------ | ------ |
| DiffPool | https://arxiv.org/abs/1806.08804 |
| SAGPool | http://proceedings.mlr.press/v97/lee19c.html |
| GAT | https://arxiv.org/abs/1710.10903 |
| g-U-Nets | http://proceedings.mlr.press/v97/gao19a.html |
| GCN | https://arxiv.org/abs/1609.02907 |
 


## Main components of our Code
| Component | Content |
| ------ | ------ |
| handle_data | Includes files required to simulate, vectorize and reshape the data. |
| reproducibility | Contains the GNN codes and the reproducibility framework implementation. |
| topogan| Contains the code files of the generative learning technique.  |

## Example Result  
![reproducibility scores](results_figure.png)
The figure demonstrates an example of output for a population of 80 subjects where each subject has 2 views (each represented by 35 by 35 matrix). We computed the reproducibility scores of 5 GNN models using two training settings (cross-validation and few-shot). For each view, we display the scores using real and generated datasets.

## Relevant References
Alaa Bessadok, Mohamed Ali Mahjoub, Islem Rekik. Brain multigraph prediction using topology-aware adversarial graph neural network. In Medical Image Analysis 72 (2021).

Nicolas George, Islem Mhiri, Islem Rekik. Identifying the best data-driven feature selection method for boosting reproducibility in classification tasks. In Pattern Recognition 101 (2020).

## YouTube video of our paper

https://youtu.be/-R1PrnX80FE

## Please cite the following paper when using our framework

```latex
@inproceedings{gharsallaoui2021,
  title={Investigating and Quantifying the Reproducibility of Graph Neural Networks in Predictive Medicine},
  author={Gharsallaoui, Mohammed Amine and Tornaci, Furkan and Rekik, Islem},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={104--116},
  year={2021},
  organization={Springer}
}
```
