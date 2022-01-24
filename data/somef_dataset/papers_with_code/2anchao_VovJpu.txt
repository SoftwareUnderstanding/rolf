# VovJpu
## Model 
>> I use the vovnet27 as the backbone to extract features.

>> -->>The vovnet27 is described in paper:An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection.the download link is :https://arxiv.org/abs/1904.09730

>> Jpu is used to get more semantic information that combine with vovnet27.

>> -->>The Jpu is described in paper:FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation.the download link is :http://export.arxiv.org/abs/1903.11816

## Model structure
>> **OSA:** is the core structure for vovnet.

![osa](images/vovnet.png)

>> **JPU:** is the core structure for FastFCN.

![jpu](images/jpu.png)

## Environment
>> If you want to implement this project.the environment need build as follow:

>>>> python==3.6 

>>>> torch==1.1.0

>>>> numpy

>>>> matplotlib

>>>> tensorboardX

## Script interpret

>> The script dataprocess.py is for data read,it's actually a iterable.

>> The script metrics.py is defined miou.

>> The script vov_jpu.py is Vovnet27 combine the Jpu.

>> The script train.py is for train the model.

## Train 
>> I trained 120 epochs.bitch size is 8.
>> when you establish the environment,then can implement this project in terminal by "python train.py"

## Attention
>> The project was completed by me independently for academic exchange. For commercial use, please contact me by email an_chao1994@163.com.
