# LambdaNetworks


## Introduction
This work builds on the foundations laid by Irwan Bello in "LambdaNetworks: Modeling long-range interactions without attention" 
[(Bello, 2021)](https://arxiv.org/abs/2102.08602). Bello proposes a method where long-range interactions are modeled by layers which 
transform contexts into linear functions called lambdas, in order to avoid the use of attention maps. The great advantage 
of lambda layers is that they require much less compute than self-attention mechanisms according to the original paper 
by Bello. This is fantastic, because it does not only provide results faster, but also saves money and has a more 
favorable carbon footprint! However, Bello still uses 32 [TPUv3s](https://cloud.google.com/tpu) and the 
200 GB sized ImageNet classification dataset. Therefore, we started this reproducibility project wondering: Could lambda 
layers be scaled to mainstream computers while keeping its attractive properties?

In 2021 the world did not only have to deal with the [COVID-19 epidemic](https://www.who.int/emergencies/diseases/novel-coronavirus-2019) 
but was struck by [chip shortages](https://www.cnbc.com/2021/02/10/whats-causing-the-chip-shortage-affecting-ps5-cars-and-more.html) 
as well due to increase in consumer electronics for working at home, shut down factories in China and the rising prices of crypto-currencies. 
This has decreased supply to record lows and prices to record highs. Resulting in a situation, whereby researchers, academics, 
and students (who are all usually on a budget) are no longer able to quickly build a cluster out of COTS (commercial off-the-shelf) 
GPUs resulting in having to deal with older, less, and less efficient hardware.

No official code was released at the time of starting the project mid-March. Therefore, in order to answer the 
aforementioned question, it is up to us to reproduce the paper by Bello as accurately as possible while trying to scale 
it down such that it can be run on an average consumer computer.



## Why lambda layers?

Lambda layers are closely related to self-attention mechanisms as they allow for modeling long-range interactions. 
However, self-attention mechanisms have a big drawback which has to do with the fact that they require attention maps 
for modeling the relative importance of layer activations, which require additional compute and are hungry for RAM 
(Random Access Memory). This makes them less applicable for use in machine vision applications which heavily rely on 
images (consisting of a grid of pixels), due to compute and RAM requirements for modeling long-range interactions 
between each of these pixels. Therefore, it is evident that this problem should be solved in order to decrease the 
training and inference times of any attention based vision task.

As Bello [(Bello, 2021)](https://arxiv.org/abs/2102.08602) says it himself in his article: "We propose _lambda layers_ which 
model long-range interactions between a query and a _structured_ set of context elements at a reduced memory 
cost. Lambda layers transform each available context into a linear function, termed a lambda, which is then directly 
applied to the corresponding query." This set of context elements is consequently summarized by the lambda layer into a
 linear function.

Linear attention mechanisms [Li et al., 2020](https://arxiv.org/abs/2007.14902) have posed a solution to the problem of high memory usage. However, 
these methods do not capture positional information between query and context elements (e.g. where pixels are on an 
image). Lambda layers, in contrast, have low memory usage and capture position information. The latter even results in 
increased performance, such that it outperforms convolutions with linear attention and local relative self-attention on 
the ImageNet dataset.

## Existing LambdaNetworks paper reviews and reproducibility projects
The paper by Bello has been published on February 2, 2021 and has been cited 7 times at the time of this work. 
This means that the article is brand new and therefore has not been combed through yet by the academic community.

However, some researchers and members of the machine learning community have already read the article and provided a
 review of the paper. 
 * [Yannic Kilcher](https://www.youtube.com/watch?v=3qxJ2WD8p4w) 
 has published "LambdaNetworks: Modeling long-range interactions without Attention (Paper Explained)" on YouTube 4 
 months prior to the publication of the article by Bello.  Kilcher goes over the preliminary version of the paper and 
 explains them to listeners and provides recommendations to the author. 
 * [Carlos Ledezma](https://www.youtube.com/watch?v=awclKwG0_sM) 
 published a video along the lines of Kilcher but takes more time to extensively clarify the distinction between the 
 structure of an attention layer and a lambda layer.
* [Phil Wang](https://github.com/lucidrains/lambda-networks) has published unofficial code using Pytorch 
about the lambda layer. 
* [Myeongjun Kim](https://github.com/leaderj1001/LambdaNetworks) has not only 
reproduced the lambda layer code, but this member of the community has also applied it to different ResNet versions
 and different datasets, as well as performing an ablation study. 
 The code from the aforementioned data scientists is not used for generating our code.

Luckily, Bello clarified that he will publish the code corresponding to the LambdaNetworks paper soon (April). This will
 most-likely enhance everyone's understanding of the LambdaNetworks.

## Our implementation
For a complete explanation of the lambda layers, their implementation and their integration within the ResNet-50 
architecture, as well as our results, conclusions and recommendations derived from this work can be found in our [blog](https://jialvear.medium.com/lambdanetworks-efficient-accurate-but-also-accessible-a-reproducibility-project-with-cifar-10-3429d8ece677). 
Besides that, we have also designed a [poster](https://github.com/joigalcar3/LambdaNetworks/blob/main/Poster.pdf) which briefly illustrates the work carried out as part of this 
reproducibility project. 

## How is this code structured
Here follows a short description of the files that make up this repository, as well as a short description of how to get 
started:
* _user\_input.py_: this file contains all the inputs that the user would like to modify
* _data\_preprocessing.py_: this file downloads the CIFAR-10 data and performs data augmentation
* _model\_preparation.py_: this file prepares the model. It selects the right model, optimizer, the criterion, learning
  rate scheduler and imports a checkpoint if necessary.
* _resnet.py_: code to generate the ResNet architectures. Code borrowed from [Pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).
* _utils.py_: file required by _resnet.py_.
* _resnet\_lambda.py_: contains the modified ResNet-50 architecture with lambda layers.
* _lambda\_layer.py_: this file contains the layer implementation of the lambda layer.
* _LabelSmoothing.py_: this file defines the label smoothing. Code borrowed from [Suvojit Manna](https://gist.github.com/suvojit-0x55aa/0afb3eefbb26d33f54e1fb9f94d6b609).
* _nntrain.py_: contains the training function.
* _nntest.py_: contains the test function.
* _log\_data.py_: data logger and printer.
* _main\_general.py_: this file contains the main loop (as a function of the epochs) and calls some of the previous 
files in the given order.

To start using the code you can download the required Python libraries stored within _requirements.txt_. For that purpose,
it is as simple as running the following command within the command line:
```shell script
pip install -r requirements.txt
```
Then adjust the parameters that you deem necessary in _user\_input.py_ and run _main\_general.py_.

You can also run it within Google Colab. For that you only copy-paste two lines. First:
```shell script
!git clone https://github.com/joigalcar3/LambdaNetworks
```
This will clone the repository. Then you can open the _user\_input.py_ file and alter the user input. Second:
```shell script
!python LambdaNetworks/main_general.py
```
And that is it. If you want to download the results of a particular run in order to run it in Tensorboard for visualisation,
 then run the following lines:
 
 ```shell script
from google.colab import files
!zip -r runs/yyyyy/xxxxx.zip runs/yyyyy/xxxxx
files.download('./runs/yyyyy/xxxxx.zip')
```
, where **yyyyy** refers to the type of model being run (Lambda or Baseline) and **xxxxx** refers to the name of the file 
within the runs/yyyyy folder that you just created.

## More info

Hope you find our work helpful for understanding the lambda layers and it will help you to integrate them in your personal 
project. Do not hesitate to reach out to us in the case that you have any questions:
* Jose Ignacio de Alvear Cardenas: [j.i.dealvearcardenas@student.tudelft.nl](mailto:j.i.dealvearcardenas@student.tudelft.nl)
* Wesley de Vries: [w.devries-1@student.tudelft.nl](mailto:w.devries-1@student.tudelft.nl)
