## From https://www.cloudops.com/2020/01/remote-sensing-data-pipelines-kubernetes-and-neural-networks-in-ecology/

The best current script is ``residual-learning.R``. It refers to data available here: https://drive.google.com/open?id=1WICa8uCge-mopWLpH2ByH8D3FMQFoO9C

The model is now much more efficient than what was described in the blog post, taking <5 epochs to converge instead of many more. This was accomplished through residual learing, with inspiration from https://arxiv.org/abs/1708.00838v1. 

I am looking to implement a UNet(https://arxiv.org/abs/1505.04597) inspired model which may be better at feature recognition and which builds upon the advances found using residual training. I am also interested in GANs to ensure the model is making realistic predictions, which is a problem for the current implementation.

```Analysis``` and ```Automated datapull``` are working directories for model construction and data download, respectively.``` Datapull``` is accomplished through a workflow based on kuberenetes pods. It should be interesting for anyone familiar with container orchestration. ```Analysis``` has poor(er) performing models and should mostly be ignored.

https://grass.osgeo.org/grass78/manuals/r.geomorphon.html. This may improve network performance.
