# cifar pytorch

WideResnet + SAM optimizer

Max test accuracy: 96.40 %

![WideResnetSam results](./results/WideResnetSam.png)

Resnet + [SAM optimizer](https://github.com/davda54/sam) ([Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)). Outputs coming from Resnet with SAM optimizer are still looking pretty [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution). [Video of training](https://www.youtube.com/watch?v=n_lHHflX-YQ)

Max test accuracy: 96.21 %

![ResnetSam results](./results/ResnetSam.png)

Original script and models from Resnet & Densenet came from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

Results for [WideResnet](https://arxiv.org/pdf/1605.07146.pdf) with model [https://github.com/davda54/sam/blob/main/example/model/wide_res_net.py](https://github.com/davda54/sam/blob/main/example/model/wide_res_net.py):

Max test accuracy: 96.16 %

![WideNet results](./results/WideResnet.png)

Results achieved with [densenet121](https://arxiv.org/abs/1608.06993):

Max test accuracy: 95.8 %

![DenseNet121 results](./results/DenseNet121.png)

Results for [RESNET18](https://arxiv.org/pdf/1512.03385.pdf):

Max test accuracy: ~ 95.2 - 95.8 % (different runs give different results a bit)

![RESNET18 results](./results/RESNET18.png)

Looking at the features coming out of resnet & creating clusters for training (upper graph) and testing (lower graph) (proc_feats.py):

![clustered features](./results/features_resnet.png)

Or in 1D (proc_feats_1d.py):

![clustered features](./results/features_resnet_1d.png)

And in 3D (proc_feats_3d.py):

![clustered features](./results/features_resnet_3d_train.png)

![clustered features](./results/features_resnet_3d_test.png)

[Video of feature evolution in 1D, 2D](https://www.youtube.com/watch?v=WbPf8EG-JnQ)

[Video of feature evolution in 1D, 2D, 3D](https://www.youtube.com/watch?v=k9tVFuk_XW4)

[Video for cross entropy loss evolution](https://www.youtube.com/watch?v=RN7T2PEjd6g)

[Video for standard deviation loss evolution](https://www.youtube.com/watch?v=CeKXZEX3_Fs)

[Video for bell's curves intersection loss evolution](https://www.youtube.com/watch?v=io-i7Vgvfq8)

Looking at features separately and their values for different classes (python3 play_feats.py), it seems like one feature is affiliated with several classes and they might be in a bit different spectrum as well (blue - train, red - test):

![feat_0](./results/feat_new_0.png)

![feat_1](./results/feat_new_1.png)

![feat_2](./results/feat_new_2.png)

![feat_3](./results/feat_new_3.png)

![feat_4](./results/feat_new_4.png)

Same for probabilities:

![probs_0](./results/probs_new_0.png)

![probs_1](./results/probs_new_1.png)

![probs_2](./results/probs_new_2.png)

Info about system:

Python 3.8.5, GCC 9.3.0, Pytorch 1.7.0+cu110, NVIDIA-SMI 455.38, Driver Version: 455.38, CUDA Version: 11.1, Ubuntu 20.04.1 LTS, GEFORCE RTX 3090


There are also other ways how to display clusters:

[t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data](https://arxiv.org/abs/1807.11824)

[sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
