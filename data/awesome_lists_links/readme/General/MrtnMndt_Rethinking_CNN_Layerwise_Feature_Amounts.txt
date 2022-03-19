# Rethinking Layer-wise Feature Amounts in Convolutional Neural Network Architectures

In this repository we provide open-source [PyTorch](https://pytorch.org) code for our NeurIPS 2018 CRACT workshop paper, where we characterize the classification accuracy of a family of VGG-like models by shifting a constant amount of total features to different convolutional layers and show how large amounts of features in early layers challenge common design assumptions. If you use or extend the code please cite our work:

If you use or extend the code please cite our work:

> Martin Mundt, Sagnik Majumder, Tobias Weis, Visvanathan Ramesh, "Rethinking Layer-wise Feature Amounts in Convolutional Neural Network Architectures", International Conference on Neural Information Processing Systems (NeurIPS) 2018, Critiquing and Correcting Trends in Machine Learning (CRACT) Workshop

You can find the complete CRACT workshop here: [https://ml-critique-correct.github.io](https://ml-critique-correct.github.io) 

and our paper here: [https://arxiv.org/abs/1812.05836](https://arxiv.org/abs/1812.05836) or through the above website. 

## Interactive visualization of results
We provide interactive visualization to explore our results more easily on the repositories github page: [https://mrtnmndt.github.io/Rethinking\_CNN\_Layerwise\_Feature\_Amounts/](https://mrtnmndt.github.io/Rethinking_CNN_Layerwise_Feature_Amounts/)

## Running the code
Our code can be used with default arguments by simply executing

`python main.py`

using python 3.5 (although the code should generally work with other python versions as well).

This will launch the CIFAR-10 experiment with approx. 200 architectural variants of the VGG-16 (D) network as specified in the paper. The dataset will be downloaded automatically and a folder per experiment created with a time stamp. All necessary parameters, ranging from datasets to hyper-parameters such as learning rate, mini-batch size as well as architectures, are exposed in the command line parser that can be found in *lib/cmdparser.py*. Note that data pre-processing (global contrast normalization) is off by default and can be added with `--preprocessing True`.

### Datasets
You can use the other datasets reported in the paper (FashionMNIST; MNIST) by adding them to the execution command. As an example: 

`python main.py --datasets MNIST`

We tried our best to use a generic data-loading pipeline by implementing classes named according to the datasets in *lib/Datasets/datasets.py*. The main file creates an instance of the dataset and dataloaders in the following lines: 

`data_init_method = getattr(datasets, args.dataset)`
`dataset = data_init_method(torch.cuda.is_available(), args)`

If you want to add another dataset, you should be able to do this by adding a class to the datasets file with appropriate dataset name and the dataset an dataloader wrapped in the PyTorch TensorDataset and DataLoaders respectively. 

### Different VGG layer amounts 
We provide a command line argument `vgg-depth` to generate VGG-like models with different amount of layers (this is possible because the 3x3 convolutions with padding = 1 do not change spatial dimensionality and can be stacked arbitrarily). We follow the pattern of the original paper, where e.g. depth = 16 corresponds to VGG-D and depth = 19 corresponds to VGG-E etc. As an example:

`python main.py --vgg-depth 19`

In theory you can add your own non-VGG like networks by adding the definition to the *lib/models/architectures.py* file. We provide convolutional and classifier blocks (with BN and batch-norm and multiple layers) for this. An interesting thing to do would be to evaluate whether our findings hold for e.g. residual networks (vgg-like networks with skip connections).

### Saving results to csv and resuming experiments
All results are getting saved automatically to a *runs/* directory with a time stamped folder. We save a csv file per architecture but keep appending previous results. This way, if the experiment crashes at some point, the run can be resumed with `resume-model-id`. 

### Single GPU usage by splitting mini-batches and automatically accumulating gradients
As many of the model variants have many filters in early layers, we have implemented a GPU memory usage estimate when the model gets build initially. Taking into account the batch size and model configuration, we automatically split the mini-batch into smaller chunks if the currently available GPU memory is exceeded. The gradients are then accumulated until the specified mini-batch size (default: 128) is reached before an optimizer update is done. While our code can be used with multiple GPUs (through use of PyTorch's DataParallel) this allows our code to be run on any single GPU. 

We note that this check crashes occasionally and memory still gets exceeded because the theoretically allocated memory and the effective memory used by CUDNN differ. We came up with a rough heuristic for this, but *if anyone using our code has a proper solution to this issue, we will very much appreciate feedback, suggestions or a pull request!*. 

### Weight initialization and learning rate schedule

As reported in the paper we make use of the *Kaiming-Normal* weight initialization of *Delving Deep into Rectifiers* by [He et. al](https://arxiv.org/abs/1502.01852) and use the learning rate schedule as proposed by [Loshchilov and Hutter](https://arxiv.org/abs/1608.03983) in *Stochastic Gradient Descent with Warm Restarts*.
Both are implemented in a `LearningRateScheduler` and `WeightInit` class. For the latter we have included options for other initialization methods that can be used. 

