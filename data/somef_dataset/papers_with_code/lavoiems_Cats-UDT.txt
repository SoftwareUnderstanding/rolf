# Integrating Semantics into Domain Translation without Supervision

Official code repository for the paper Integrating Semantics into Domain Translation without Supervision.

Dependencies are provided in `requirement.txt`. A Dockerfile is provided for reproducing the same environment
used to run the experiments.

## Organization

The models are in `src/models`. Every model has 3 files:

`__init__.py`: Defines the specific parameters of the models

`model.py` Defines the architecture of the model

`train.py` Defines the training algorithm of the model

and a folder `evaluate` containing the evaluation scripts.

In general, a model can be trained by invoking `main.py`, which also contains the general parameters shared among all the
models. The syntax for training a model is as follows:
```
python src/main.py [GENERAL PARAMETERS] [MODEL NAME] [SPECIFIC MODEL PARAMETERS]
```
Some models can be evaluated by using one of it's evaluation script in the folder `evaluate`.
The syntax for evaluating a model is a follows:
```
python src/evaluate.py [MODEL NAME]-[EVALUATE SCRIPT NAME] [SPECIFIC EVALUATIONS PARAMETERS]
```
## Datasets

We presented results on the MNIST-SVHN dataset which can be themselves downloaded by the torchvision library. We also
presented results on the Sketch->Real dataset which are a subset of the DomainNet dataset
(https://arxiv.org/abs/1812.01754).
The sketch dataset can be downloaded as follow:
```bash
./download_data.sh sketch
```
Similarly, the real dataset can be downloaded as follows:
```bash
./download_data.sh real
```

## Models
This repository is composed of the models that are useful for reproducing the results from the paper.
We now explain how to train them.
### classifier
This model is used for evaluating the translation MNIST<->SVHN. The classifier is a wide residual network
(https://arxiv.org/abs/1605.07146) and the is code inspired from:
https://github.com/szagoruyko/wide-residual-networks.

**Classifying MNIST**
```bash
python src/main.py --exp-name classifier --cuda --run-id mnist --train-batch-size 128 --valid-split 0.2 classifier --dataset mnist
```
**Classifying SVHN**
```bash
python src/main.py --exp-name classifier --cuda --run-id svhn --train-batch-size 128 --valid-split 0.2 classifier --dataset svhn_extra
```
### imsat
Imsat (https://arxiv.org/abs/1605.07146) is a method for clustering using deep neural networks. In this work, we use it
 for clustering MNIST. We use a slighly different version than the one proposed in the original model. We explain the
 the differences in the appendix on the paper. But, the original imsat algorithm, or any other cluster algorithm which
 can cluster MNIST decently well could be used in place.

**Clustering MNIST**
```bash
python src/main.py --exp-name cluster --cuda --run-id mnist imsat
```

### vmt_cluster
VMT (https://arxiv.org/abs/1905.04215) is an unsupervised domain adaptation method. In this work, we proposed to replace
the ground truth labels of the source domain with the learned cluster on the source domain. `vmt_cluster` is a model
that applies the vmt method on images by considering the learned cluster rather than the ground truth labels.

**Domain adaptation with clustering MNIST-SVHN**
```bash
python src/main.py --exp-name vmt-cluster --cuda --run-id mnist-svhn vmt_cluster --dataset1 mnist --dataset2 svhn --cluster-model-path ./experiments/imsat/cluster_mnist-None --dw 0.01 --svw 1 --tvw 0.06 --tcw 0.06 --smw 1 --tmw 0.06
```
### vmtc_repr
We also propose to apply VMT cluster on the representation learned by a representation learning model using the clusters
learned themselves on the representation. In this work, we used the pre-trained MoCO-v2
(https://arxiv.org/abs/2003.04297) model, which can be downloaded as follow
```bash
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
```
We found that using this method for learning cross-domain clustering on sketch-real yielded better results. Learning the
model can be done as follows:

**Domain adaptation with clustering on Sketch-Real**
```bash
python src/main.py --exp-name vmtc-repr --cuda --run-id sketch-real vmtc_repr --ss-path moco_v2_800ep_pretrain.pth.tar
```

### sg_sem
We propose to use the learned semantics in a domain translation framework. In this work, we modified StarGAN-V2. The
same framework can be used for both MNIST<->SVHN and Sketches->Reals.

**Domain translation MNIST<->SVHN**
```bash
python src/main.py --exp-name sg-sem --run-id mnist-svhn --cuda --train-batch-size 8 sg_sem --num_classes 10 --img_size 32 --dataset_loc data --dataset cond_mnist_svhn --cluster_type vmt_cluster --cluster_path experiments/vmt_cluster/vmt-cluster_mnist-svhn-None/
```

**Domain translation Sketches->Reals**
```bash
python src/main.py --exp-name sg_sem --run-id sketch-real --cuda --train-batch-size 8 sg_sem --dataset_loc data --dataset cond_visda --lambda_sty 0 --ss_path moco_v2_800ep_pretrain.pth.tar --cluster_path experiments/vmtc_repr/vmtc-repr_sketch-real-None/
```


#### Fetch the results
To create the MNIST->SVHN grids, simply run the following command:
```bash
python src/evaluate.py sg_sem-fetch_results_ms --state-dict-path experiments/sg-sem_mnist-svhn-None/model/nets_ema:100000.ckpt  --data-root-src data --dataset-src svhn --domain 0 --da-path ./experiments/vmt_cluster/vmt-cluster_mnist-svhn-None/ --save-name MNIST-SVHN
```
To create the Sketches->Reals grid, simply run the following command:
```bash
python src/evaluate.py sg_sem-fetch_results_sr --state-dict-path experiments/sg-sem_sketch-real-None/model/nets_ema:100000.ckpt --data-root-src data/test_all/sketch --domain 0 --ss-path moco_v2_800ep_pretrain.pth.tar --da-path experiments/vmtc_repr/vmtc-repr_sketch-real-None/ --save-name Sketch-Real --model-path experiments/sg-sem_sketch-real-None
```

#### Fetch the results
To fetch results, use the script `fetch_results.py`, provided. Similarly, one can compute the FID using the script `compute_fid.py`.


## Results Unsupervised Domain Translation.

**Sketch->Real**

![](assets/ours_sr.png)


**MNIST->SVHN**

![](assets/ours_ms.png)

