# Spatial Transformer Networks

Augment your network using a visual attention mechanism called spatial  
transformer networks.  

Spatial transformer networks are a generalization of differentiable  
attention to any spatial transformation. Spatial transformer networks  
(STN for short) allow a neural network to learn how to perform spatial  
transformations on the input image in order to enhance the geometric   
invariance of the model.  
For example, it can crop a region of interest, scale and correct  
the orientation of an image. It can be a useful mechanism because CNNs  
are not invariant to rotation and scale and more general affine  
transformations.  

One of the best things about STN is the ability to simply plug it into  
any existing CNN with very little modification.  

## Depicting spatial transformer networks
--------------------------------------

Spatial transformer networks boils down to three main components :  

-  The localization network is a regular CNN which regresses the  
   transformation parameters. The transformation is never learned  
   explicitly from this dataset, instead the network learns automatically  
   the spatial transformations that enhances the global accuracy.  
-  The grid generator generates a grid of coordinates in the input  
   image corresponding to each pixel from the output image.  
-  The sampler uses the parameters of the transformation and applies  
   it to the input image.  

## papers

spatial transformer networks in the DeepMind paper <https://arxiv.org/abs/1506.02025>  

## dataset

mnist

## how to run?

```shell
bash run.sh
```

## output
compare  
![Alt](./output/compare.jpg)
epoch acc  
![Alt](./output/epoch_acc.jpg)
epoch loss  
![Alt](./output/epoch_loss.jpg)
step acc  
![Alt](./output/step_acc.jpg)
step loss  
![Alt](./output/step_loss.jpg)
output  
```shell
Net(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(1, 8, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace)
    (3): Conv2d(8, 10, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=90, out_features=32, bias=True)
    (1): ReLU(inplace)
    (2): Linear(in_features=32, out_features=6, bias=True)
  )
)
config:
early_stop_epoch_limit : 5
loss : NLL
momentum : 0.9
print_every : 100
epochs : 100
train_load_check_point_file : True
num_workers : 4
learn_rate : 0.01
batch_size : 128
early_stop_epoch : True
early_stop_step_limit : 2000
data_path : ./raw
early_stop_step : True
epoch_only : True
device : cuda
optimizer : SGD
dataset : mnist
[1/99] [Step:100] [Train Loss:2.167608 Acc:0.234375 12672/60000 (21%)] [Val Loss:2.086288 Acc:0.447800 4478/10000 (45%)] [Best Epoch:1 Loss:2.086288 Acc:0.447800] [Best Step:100 Loss:2.086288 Acc:0.447800]
[1/99] [Step:200] [Train Loss:0.650624 Acc:0.820312 25472/60000 (42%)] [Val Loss:0.641685 Acc:0.832100 8321/10000 (83%)] [Best Epoch:1 Loss:0.641685 Acc:0.832100] [Best Step:200 Loss:0.641685 Acc:0.832100]
[1/99] [Step:300] [Train Loss:0.447366 Acc:0.828125 38272/60000 (64%)] [Val Loss:0.409871 Acc:0.871000 8710/10000 (87%)] [Best Epoch:1 Loss:0.409871 Acc:0.871000] [Best Step:300 Loss:0.409871 Acc:0.871000]
[1/99] [Step:400] [Train Loss:0.260484 Acc:0.929688 51072/60000 (85%)] [Val Loss:0.286122 Acc:0.917600 9176/10000 (92%)] [Best Epoch:1 Loss:0.286122 Acc:0.917600] [Best Step:400 Loss:0.286122 Acc:0.917600]
[1/99] [Step:400] [Train Loss:1.009604 Acc:0.678855] [Val Loss:0.255957 Acc:0.926300 9263/10000 (93%)] [Best Epoch:1 Loss:0.255957 Acc:0.926300] [Best Step:400 Loss:0.255957 Acc:0.926300] [3.38s 3.4s]
[2/99] [Step:500] [Train Loss:0.719501 Acc:0.742188 12672/60000 (21%)] [Val Loss:0.325823 Acc:0.928400 9284/10000 (93%)] [Best Epoch:1 Loss:0.255957 Acc:0.926300] [Best Step:400 Loss:0.255957 Acc:0.926300]
[2/99] [Step:600] [Train Loss:0.145961 Acc:0.968750 25472/60000 (42%)] [Val Loss:0.197905 Acc:0.943400 9434/10000 (94%)] [Best Epoch:2 Loss:0.197905 Acc:0.943400] [Best Step:600 Loss:0.197905 Acc:0.943400]
[2/99] [Step:700] [Train Loss:0.214439 Acc:0.929688 38272/60000 (64%)] [Val Loss:0.152496 Acc:0.957600 9576/10000 (96%)] [Best Epoch:2 Loss:0.152496 Acc:0.957600] [Best Step:700 Loss:0.152496 Acc:0.957600]
[2/99] [Step:800] [Train Loss:0.171293 Acc:0.937500 51072/60000 (85%)] [Val Loss:0.179137 Acc:0.946500 9465/10000 (95%)] [Best Epoch:2 Loss:0.152496 Acc:0.957600] [Best Step:700 Loss:0.152496 Acc:0.957600]
[2/99] [Step:800] [Train Loss:0.352962 Acc:0.887743] [Val Loss:0.155159 Acc:0.956600 9566/10000 (96%)] [Best Epoch:2 Loss:0.152496 Acc:0.957600] [Best Step:700 Loss:0.152496 Acc:0.957600] [3.30s 6.7s]
[3/99] [Step:900] [Train Loss:0.718812 Acc:0.781250 12672/60000 (21%)] [Val Loss:0.185195 Acc:0.958100 9581/10000 (96%)] [Best Epoch:2 Loss:0.152496 Acc:0.957600] [Best Step:700 Loss:0.152496 Acc:0.957600]
[3/99] [Step:1000] [Train Loss:0.147973 Acc:0.945312 25472/60000 (42%)] [Val Loss:0.122857 Acc:0.964500 9645/10000 (96%)] [Best Epoch:3 Loss:0.122857 Acc:0.964500] [Best Step:1000 Loss:0.122857 Acc:0.964500]
[3/99] [Step:1100] [Train Loss:0.071183 Acc:0.976562 38272/60000 (64%)] [Val Loss:0.116625 Acc:0.965800 9658/10000 (97%)] [Best Epoch:3 Loss:0.116625 Acc:0.965800] [Best Step:1100 Loss:0.116625 Acc:0.965800]
[3/99] [Step:1200] [Train Loss:0.092998 Acc:0.960938 51072/60000 (85%)] [Val Loss:0.111858 Acc:0.967000 9670/10000 (97%)] [Best Epoch:3 Loss:0.111858 Acc:0.967000] [Best Step:1200 Loss:0.111858 Acc:0.967000]
[3/99] [Step:1200] [Train Loss:0.248452 Acc:0.922291] [Val Loss:0.101634 Acc:0.970500 9705/10000 (97%)] [Best Epoch:3 Loss:0.101634 Acc:0.970500] [Best Step:1200 Loss:0.101634 Acc:0.970500] [3.40s 10.1s]
[4/99] [Step:1300] [Train Loss:0.388234 Acc:0.875000 12672/60000 (21%)] [Val Loss:0.141787 Acc:0.961300 9613/10000 (96%)] [Best Epoch:3 Loss:0.101634 Acc:0.970500] [Best Step:1200 Loss:0.101634 Acc:0.970500]
[4/99] [Step:1400] [Train Loss:0.131049 Acc:0.960938 25472/60000 (42%)] [Val Loss:0.115066 Acc:0.965600 9656/10000 (97%)] [Best Epoch:3 Loss:0.101634 Acc:0.970500] [Best Step:1200 Loss:0.101634 Acc:0.970500]
[4/99] [Step:1500] [Train Loss:0.126288 Acc:0.968750 38272/60000 (64%)] [Val Loss:0.090482 Acc:0.973000 9730/10000 (97%)] [Best Epoch:4 Loss:0.090482 Acc:0.973000] [Best Step:1500 Loss:0.090482 Acc:0.973000]
[4/99] [Step:1600] [Train Loss:0.121875 Acc:0.968750 51072/60000 (85%)] [Val Loss:0.099986 Acc:0.971000 9710/10000 (97%)] [Best Epoch:4 Loss:0.090482 Acc:0.973000] [Best Step:1500 Loss:0.090482 Acc:0.973000]
[4/99] [Step:1600] [Train Loss:0.199811 Acc:0.937300] [Val Loss:0.086782 Acc:0.973000 9730/10000 (97%)] [Best Epoch:4 Loss:0.086782 Acc:0.973000] [Best Step:1600 Loss:0.086782 Acc:0.973000] [3.40s 13.5s]
[5/99] [Step:1700] [Train Loss:0.380366 Acc:0.882812 12672/60000 (21%)] [Val Loss:0.131906 Acc:0.962100 9621/10000 (96%)] [Best Epoch:4 Loss:0.086782 Acc:0.973000] [Best Step:1600 Loss:0.086782 Acc:0.973000]
[5/99] [Step:1800] [Train Loss:0.131178 Acc:0.976562 25472/60000 (42%)] [Val Loss:0.089699 Acc:0.973100 9731/10000 (97%)] [Best Epoch:4 Loss:0.086782 Acc:0.973000] [Best Step:1600 Loss:0.086782 Acc:0.973000]
[5/99] [Step:1900] [Train Loss:0.068963 Acc:0.953125 38272/60000 (64%)] [Val Loss:0.086435 Acc:0.973900 9739/10000 (97%)] [Best Epoch:5 Loss:0.086435 Acc:0.973900] [Best Step:1900 Loss:0.086435 Acc:0.973900]
[5/99] [Step:2000] [Train Loss:0.112777 Acc:0.953125 51072/60000 (85%)] [Val Loss:0.075983 Acc:0.977200 9772/10000 (98%)] [Best Epoch:5 Loss:0.075983 Acc:0.977200] [Best Step:2000 Loss:0.075983 Acc:0.977200]
[5/99] [Step:2000] [Train Loss:0.186322 Acc:0.943530] [Val Loss:0.097913 Acc:0.970700 9707/10000 (97%)] [Best Epoch:5 Loss:0.075983 Acc:0.977200] [Best Step:2000 Loss:0.075983 Acc:0.977200] [3.39s 16.9s]
[6/99] [Step:2100] [Train Loss:0.423293 Acc:0.875000 12672/60000 (21%)] [Val Loss:0.112498 Acc:0.968800 9688/10000 (97%)] [Best Epoch:5 Loss:0.075983 Acc:0.977200] [Best Step:2000 Loss:0.075983 Acc:0.977200]
[6/99] [Step:2200] [Train Loss:0.096333 Acc:0.968750 25472/60000 (42%)] [Val Loss:0.083600 Acc:0.975500 9755/10000 (98%)] [Best Epoch:5 Loss:0.075983 Acc:0.977200] [Best Step:2000 Loss:0.075983 Acc:0.977200]
[6/99] [Step:2300] [Train Loss:0.053793 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.081742 Acc:0.974600 9746/10000 (97%)] [Best Epoch:5 Loss:0.075983 Acc:0.977200] [Best Step:2000 Loss:0.075983 Acc:0.977200]
[6/99] [Step:2400] [Train Loss:0.027783 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.073726 Acc:0.978200 9782/10000 (98%)] [Best Epoch:6 Loss:0.073726 Acc:0.978200] [Best Step:2400 Loss:0.073726 Acc:0.978200]
[6/99] [Step:2400] [Train Loss:0.167590 Acc:0.947395] [Val Loss:0.073935 Acc:0.977100 9771/10000 (98%)] [Best Epoch:6 Loss:0.073726 Acc:0.978200] [Best Step:2400 Loss:0.073726 Acc:0.978200] [3.45s 20.3s]
[7/99] [Step:2500] [Train Loss:0.290204 Acc:0.906250 12672/60000 (21%)] [Val Loss:0.085232 Acc:0.976000 9760/10000 (98%)] [Best Epoch:6 Loss:0.073726 Acc:0.978200] [Best Step:2400 Loss:0.073726 Acc:0.978200]
[7/99] [Step:2600] [Train Loss:0.094800 Acc:0.968750 25472/60000 (42%)] [Val Loss:0.075008 Acc:0.977100 9771/10000 (98%)] [Best Epoch:6 Loss:0.073726 Acc:0.978200] [Best Step:2400 Loss:0.073726 Acc:0.978200]
[7/99] [Step:2700] [Train Loss:0.039663 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.067499 Acc:0.979900 9799/10000 (98%)] [Best Epoch:7 Loss:0.067499 Acc:0.979900] [Best Step:2700 Loss:0.067499 Acc:0.979900]
[7/99] [Step:2800] [Train Loss:0.128573 Acc:0.960938 51072/60000 (85%)] [Val Loss:0.064232 Acc:0.981200 9812/10000 (98%)] [Best Epoch:7 Loss:0.064232 Acc:0.981200] [Best Step:2800 Loss:0.064232 Acc:0.981200]
[7/99] [Step:2800] [Train Loss:0.126915 Acc:0.960238] [Val Loss:0.072058 Acc:0.978800 9788/10000 (98%)] [Best Epoch:7 Loss:0.064232 Acc:0.981200] [Best Step:2800 Loss:0.064232 Acc:0.981200] [3.37s 23.7s]
[8/99] [Step:2900] [Train Loss:0.425418 Acc:0.875000 12672/60000 (21%)] [Val Loss:0.088375 Acc:0.975400 9754/10000 (98%)] [Best Epoch:7 Loss:0.064232 Acc:0.981200] [Best Step:2800 Loss:0.064232 Acc:0.981200]
[8/99] [Step:3000] [Train Loss:0.087683 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.065449 Acc:0.981800 9818/10000 (98%)] [Best Epoch:7 Loss:0.064232 Acc:0.981200] [Best Step:2800 Loss:0.064232 Acc:0.981200]
[8/99] [Step:3100] [Train Loss:0.111097 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.069611 Acc:0.978100 9781/10000 (98%)] [Best Epoch:7 Loss:0.064232 Acc:0.981200] [Best Step:2800 Loss:0.064232 Acc:0.981200]
[8/99] [Step:3200] [Train Loss:0.022433 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.061153 Acc:0.980300 9803/10000 (98%)] [Best Epoch:8 Loss:0.061153 Acc:0.980300] [Best Step:3200 Loss:0.061153 Acc:0.980300]
[8/99] [Step:3200] [Train Loss:0.120913 Acc:0.962503] [Val Loss:0.060698 Acc:0.981300 9813/10000 (98%)] [Best Epoch:8 Loss:0.060698 Acc:0.981300] [Best Step:3200 Loss:0.060698 Acc:0.981300] [3.35s 27.0s]
[9/99] [Step:3300] [Train Loss:0.291002 Acc:0.906250 12672/60000 (21%)] [Val Loss:0.083161 Acc:0.974200 9742/10000 (97%)] [Best Epoch:8 Loss:0.060698 Acc:0.981300] [Best Step:3200 Loss:0.060698 Acc:0.981300]
[9/99] [Step:3400] [Train Loss:0.037885 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.063314 Acc:0.980400 9804/10000 (98%)] [Best Epoch:8 Loss:0.060698 Acc:0.981300] [Best Step:3200 Loss:0.060698 Acc:0.981300]
[9/99] [Step:3500] [Train Loss:0.088513 Acc:0.960938 38272/60000 (64%)] [Val Loss:0.066035 Acc:0.979500 9795/10000 (98%)] [Best Epoch:8 Loss:0.060698 Acc:0.981300] [Best Step:3200 Loss:0.060698 Acc:0.981300]
[9/99] [Step:3600] [Train Loss:0.076196 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.060453 Acc:0.981600 9816/10000 (98%)] [Best Epoch:9 Loss:0.060453 Acc:0.981600] [Best Step:3600 Loss:0.060453 Acc:0.981600]
[9/99] [Step:3600] [Train Loss:0.111194 Acc:0.965768] [Val Loss:0.056770 Acc:0.983100 9831/10000 (98%)] [Best Epoch:9 Loss:0.056770 Acc:0.983100] [Best Step:3600 Loss:0.056770 Acc:0.983100] [3.48s 30.5s]
[10/99] [Step:3700] [Train Loss:0.169361 Acc:0.960938 12672/60000 (21%)] [Val Loss:0.065783 Acc:0.980100 9801/10000 (98%)] [Best Epoch:9 Loss:0.056770 Acc:0.983100] [Best Step:3600 Loss:0.056770 Acc:0.983100]
[10/99] [Step:3800] [Train Loss:0.081021 Acc:0.976562 25472/60000 (42%)] [Val Loss:0.063623 Acc:0.981400 9814/10000 (98%)] [Best Epoch:9 Loss:0.056770 Acc:0.983100] [Best Step:3600 Loss:0.056770 Acc:0.983100]
[10/99] [Step:3900] [Train Loss:0.023760 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.058074 Acc:0.982700 9827/10000 (98%)] [Best Epoch:9 Loss:0.056770 Acc:0.983100] [Best Step:3600 Loss:0.056770 Acc:0.983100]
[10/99] [Step:4000] [Train Loss:0.057676 Acc:0.968750 51072/60000 (85%)] [Val Loss:0.053754 Acc:0.983400 9834/10000 (98%)] [Best Epoch:10 Loss:0.053754 Acc:0.983400] [Best Step:4000 Loss:0.053754 Acc:0.983400]
[10/99] [Step:4000] [Train Loss:0.114333 Acc:0.965119] [Val Loss:0.054047 Acc:0.983200 9832/10000 (98%)] [Best Epoch:10 Loss:0.053754 Acc:0.983400] [Best Step:4000 Loss:0.053754 Acc:0.983400] [3.39s 33.9s]
[11/99] [Step:4100] [Train Loss:0.294036 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.061702 Acc:0.981100 9811/10000 (98%)] [Best Epoch:10 Loss:0.053754 Acc:0.983400] [Best Step:4000 Loss:0.053754 Acc:0.983400]
[11/99] [Step:4200] [Train Loss:0.024356 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.054516 Acc:0.982300 9823/10000 (98%)] [Best Epoch:10 Loss:0.053754 Acc:0.983400] [Best Step:4000 Loss:0.053754 Acc:0.983400]
[11/99] [Step:4300] [Train Loss:0.055912 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.053751 Acc:0.983700 9837/10000 (98%)] [Best Epoch:11 Loss:0.053751 Acc:0.983700] [Best Step:4300 Loss:0.053751 Acc:0.983700]
[11/99] [Step:4400] [Train Loss:0.017414 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.055410 Acc:0.983100 9831/10000 (98%)] [Best Epoch:11 Loss:0.053751 Acc:0.983700] [Best Step:4300 Loss:0.053751 Acc:0.983700]
[11/99] [Step:4400] [Train Loss:0.095730 Acc:0.970416] [Val Loss:0.051519 Acc:0.983900 9839/10000 (98%)] [Best Epoch:11 Loss:0.051519 Acc:0.983900] [Best Step:4400 Loss:0.051519 Acc:0.983900] [3.44s 37.4s]
[12/99] [Step:4500] [Train Loss:0.389634 Acc:0.898438 12672/60000 (21%)] [Val Loss:0.069784 Acc:0.980100 9801/10000 (98%)] [Best Epoch:11 Loss:0.051519 Acc:0.983900] [Best Step:4400 Loss:0.051519 Acc:0.983900]
[12/99] [Step:4600] [Train Loss:0.030134 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.053168 Acc:0.982800 9828/10000 (98%)] [Best Epoch:11 Loss:0.051519 Acc:0.983900] [Best Step:4400 Loss:0.051519 Acc:0.983900]
[12/99] [Step:4700] [Train Loss:0.072564 Acc:0.976562 38272/60000 (64%)] [Val Loss:0.053659 Acc:0.983000 9830/10000 (98%)] [Best Epoch:11 Loss:0.051519 Acc:0.983900] [Best Step:4400 Loss:0.051519 Acc:0.983900]
[12/99] [Step:4800] [Train Loss:0.060457 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.049466 Acc:0.984600 9846/10000 (98%)] [Best Epoch:12 Loss:0.049466 Acc:0.984600] [Best Step:4800 Loss:0.049466 Acc:0.984600]
[12/99] [Step:4800] [Train Loss:0.090151 Acc:0.972248] [Val Loss:0.051032 Acc:0.984700 9847/10000 (98%)] [Best Epoch:12 Loss:0.049466 Acc:0.984600] [Best Step:4800 Loss:0.049466 Acc:0.984600] [3.40s 40.8s]
[13/99] [Step:4900] [Train Loss:0.272001 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.052918 Acc:0.984300 9843/10000 (98%)] [Best Epoch:12 Loss:0.049466 Acc:0.984600] [Best Step:4800 Loss:0.049466 Acc:0.984600]
[13/99] [Step:5000] [Train Loss:0.045340 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.049422 Acc:0.984500 9845/10000 (98%)] [Best Epoch:13 Loss:0.049422 Acc:0.984500] [Best Step:5000 Loss:0.049422 Acc:0.984500]
[13/99] [Step:5100] [Train Loss:0.007476 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.056097 Acc:0.982700 9827/10000 (98%)] [Best Epoch:13 Loss:0.049422 Acc:0.984500] [Best Step:5000 Loss:0.049422 Acc:0.984500]
[13/99] [Step:5200] [Train Loss:0.059180 Acc:0.976562 51072/60000 (85%)] [Val Loss:0.049353 Acc:0.984100 9841/10000 (98%)] [Best Epoch:13 Loss:0.049353 Acc:0.984100] [Best Step:5200 Loss:0.049353 Acc:0.984100]
[13/99] [Step:5200] [Train Loss:0.085826 Acc:0.973081] [Val Loss:0.048486 Acc:0.985600 9856/10000 (99%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600] [3.34s 44.1s]
[14/99] [Step:5300] [Train Loss:0.172275 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.053440 Acc:0.983600 9836/10000 (98%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600]
[14/99] [Step:5400] [Train Loss:0.012319 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.049988 Acc:0.984600 9846/10000 (98%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600]
[14/99] [Step:5500] [Train Loss:0.020327 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.051801 Acc:0.983500 9835/10000 (98%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600]
[14/99] [Step:5600] [Train Loss:0.135970 Acc:0.953125 51072/60000 (85%)] [Val Loss:0.056238 Acc:0.982700 9827/10000 (98%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600]
[14/99] [Step:5600] [Train Loss:0.081670 Acc:0.974230] [Val Loss:0.049181 Acc:0.985400 9854/10000 (99%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600] [3.37s 47.5s]
[15/99] [Step:5700] [Train Loss:0.226850 Acc:0.945312 12672/60000 (21%)] [Val Loss:0.053288 Acc:0.984400 9844/10000 (98%)] [Best Epoch:13 Loss:0.048486 Acc:0.985600] [Best Step:5200 Loss:0.048486 Acc:0.985600]
[15/99] [Step:5800] [Train Loss:0.006131 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.047086 Acc:0.986500 9865/10000 (99%)] [Best Epoch:15 Loss:0.047086 Acc:0.986500] [Best Step:5800 Loss:0.047086 Acc:0.986500]
[15/99] [Step:5900] [Train Loss:0.034826 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.046471 Acc:0.986000 9860/10000 (99%)] [Best Epoch:15 Loss:0.046471 Acc:0.986000] [Best Step:5900 Loss:0.046471 Acc:0.986000]
[15/99] [Step:6000] [Train Loss:0.017590 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.047672 Acc:0.985700 9857/10000 (99%)] [Best Epoch:15 Loss:0.046471 Acc:0.986000] [Best Step:5900 Loss:0.046471 Acc:0.986000]
[15/99] [Step:6000] [Train Loss:0.078895 Acc:0.976013] [Val Loss:0.049697 Acc:0.985200 9852/10000 (99%)] [Best Epoch:15 Loss:0.046471 Acc:0.986000] [Best Step:5900 Loss:0.046471 Acc:0.986000] [3.38s 50.8s]
[16/99] [Step:6100] [Train Loss:0.290664 Acc:0.914062 12672/60000 (21%)] [Val Loss:0.082496 Acc:0.976600 9766/10000 (98%)] [Best Epoch:15 Loss:0.046471 Acc:0.986000] [Best Step:5900 Loss:0.046471 Acc:0.986000]
[16/99] [Step:6200] [Train Loss:0.024184 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.050338 Acc:0.983800 9838/10000 (98%)] [Best Epoch:15 Loss:0.046471 Acc:0.986000] [Best Step:5900 Loss:0.046471 Acc:0.986000]
[16/99] [Step:6300] [Train Loss:0.030814 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.045457 Acc:0.985800 9858/10000 (99%)] [Best Epoch:16 Loss:0.045457 Acc:0.985800] [Best Step:6300 Loss:0.045457 Acc:0.985800]
[16/99] [Step:6400] [Train Loss:0.026612 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.042742 Acc:0.986600 9866/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[16/99] [Step:6400] [Train Loss:0.089405 Acc:0.973114] [Val Loss:0.046101 Acc:0.984500 9845/10000 (98%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600] [3.40s 54.2s]
[17/99] [Step:6500] [Train Loss:0.179858 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.066143 Acc:0.980600 9806/10000 (98%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[17/99] [Step:6600] [Train Loss:0.051735 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.048146 Acc:0.985200 9852/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[17/99] [Step:6700] [Train Loss:0.072117 Acc:0.976562 38272/60000 (64%)] [Val Loss:0.043320 Acc:0.986200 9862/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[17/99] [Step:6800] [Train Loss:0.079231 Acc:0.976562 51072/60000 (85%)] [Val Loss:0.045918 Acc:0.985700 9857/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[17/99] [Step:6800] [Train Loss:0.077812 Acc:0.976496] [Val Loss:0.045088 Acc:0.985500 9855/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600] [3.39s 57.6s]
[18/99] [Step:6900] [Train Loss:0.174333 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.049611 Acc:0.984400 9844/10000 (98%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[18/99] [Step:7000] [Train Loss:0.009792 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.042843 Acc:0.986000 9860/10000 (99%)] [Best Epoch:16 Loss:0.042742 Acc:0.986600] [Best Step:6400 Loss:0.042742 Acc:0.986600]
[18/99] [Step:7100] [Train Loss:0.033177 Acc:0.976562 38272/60000 (64%)] [Val Loss:0.042474 Acc:0.986700 9867/10000 (99%)] [Best Epoch:18 Loss:0.042474 Acc:0.986700] [Best Step:7100 Loss:0.042474 Acc:0.986700]
[18/99] [Step:7200] [Train Loss:0.012799 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.043784 Acc:0.986400 9864/10000 (99%)] [Best Epoch:18 Loss:0.042474 Acc:0.986700] [Best Step:7100 Loss:0.042474 Acc:0.986700]
[18/99] [Step:7200] [Train Loss:0.071787 Acc:0.977679] [Val Loss:0.042041 Acc:0.987300 9873/10000 (99%)] [Best Epoch:18 Loss:0.042041 Acc:0.987300] [Best Step:7200 Loss:0.042041 Acc:0.987300] [3.42s 61.1s]
[19/99] [Step:7300] [Train Loss:0.113954 Acc:0.968750 12672/60000 (21%)] [Val Loss:0.048937 Acc:0.985200 9852/10000 (99%)] [Best Epoch:18 Loss:0.042041 Acc:0.987300] [Best Step:7200 Loss:0.042041 Acc:0.987300]
[19/99] [Step:7400] [Train Loss:0.042720 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.044072 Acc:0.986200 9862/10000 (99%)] [Best Epoch:18 Loss:0.042041 Acc:0.987300] [Best Step:7200 Loss:0.042041 Acc:0.987300]
[19/99] [Step:7500] [Train Loss:0.061433 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.044358 Acc:0.987100 9871/10000 (99%)] [Best Epoch:18 Loss:0.042041 Acc:0.987300] [Best Step:7200 Loss:0.042041 Acc:0.987300]
[19/99] [Step:7600] [Train Loss:0.029867 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.043359 Acc:0.986600 9866/10000 (99%)] [Best Epoch:18 Loss:0.042041 Acc:0.987300] [Best Step:7200 Loss:0.042041 Acc:0.987300]
[19/99] [Step:7600] [Train Loss:0.068573 Acc:0.978495] [Val Loss:0.041649 Acc:0.987300 9873/10000 (99%)] [Best Epoch:19 Loss:0.041649 Acc:0.987300] [Best Step:7600 Loss:0.041649 Acc:0.987300] [3.43s 64.5s]
[20/99] [Step:7700] [Train Loss:0.244270 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.057400 Acc:0.983100 9831/10000 (98%)] [Best Epoch:19 Loss:0.041649 Acc:0.987300] [Best Step:7600 Loss:0.041649 Acc:0.987300]
[20/99] [Step:7800] [Train Loss:0.029652 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.040923 Acc:0.986900 9869/10000 (99%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900]
[20/99] [Step:7900] [Train Loss:0.024100 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.043758 Acc:0.987000 9870/10000 (99%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900]
[20/99] [Step:8000] [Train Loss:0.031634 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.041608 Acc:0.987400 9874/10000 (99%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900]
[20/99] [Step:8000] [Train Loss:0.067934 Acc:0.978778] [Val Loss:0.043122 Acc:0.986700 9867/10000 (99%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900] [3.35s 67.8s]
[21/99] [Step:8100] [Train Loss:0.185643 Acc:0.929688 12672/60000 (21%)] [Val Loss:0.049858 Acc:0.985400 9854/10000 (99%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900]
[21/99] [Step:8200] [Train Loss:0.047171 Acc:0.976562 25472/60000 (42%)] [Val Loss:0.052444 Acc:0.983800 9838/10000 (98%)] [Best Epoch:20 Loss:0.040923 Acc:0.986900] [Best Step:7800 Loss:0.040923 Acc:0.986900]
[21/99] [Step:8300] [Train Loss:0.005207 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.040902 Acc:0.987200 9872/10000 (99%)] [Best Epoch:21 Loss:0.040902 Acc:0.987200] [Best Step:8300 Loss:0.040902 Acc:0.987200]
[21/99] [Step:8400] [Train Loss:0.043039 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.043793 Acc:0.985700 9857/10000 (99%)] [Best Epoch:21 Loss:0.040902 Acc:0.987200] [Best Step:8300 Loss:0.040902 Acc:0.987200]
[21/99] [Step:8400] [Train Loss:0.066089 Acc:0.979628] [Val Loss:0.040721 Acc:0.986300 9863/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300] [3.40s 71.2s]
[22/99] [Step:8500] [Train Loss:0.249187 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.046625 Acc:0.986700 9867/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[22/99] [Step:8600] [Train Loss:0.037794 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.042668 Acc:0.986700 9867/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[22/99] [Step:8700] [Train Loss:0.007556 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.041184 Acc:0.986100 9861/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[22/99] [Step:8800] [Train Loss:0.047617 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.045022 Acc:0.986000 9860/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[22/99] [Step:8800] [Train Loss:0.064390 Acc:0.979727] [Val Loss:0.042038 Acc:0.987100 9871/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300] [3.42s 74.7s]
[23/99] [Step:8900] [Train Loss:0.172490 Acc:0.953125 12672/60000 (21%)] [Val Loss:0.051839 Acc:0.985000 9850/10000 (98%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[23/99] [Step:9000] [Train Loss:0.099852 Acc:0.968750 25472/60000 (42%)] [Val Loss:0.043654 Acc:0.986400 9864/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[23/99] [Step:9100] [Train Loss:0.033233 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.047484 Acc:0.985200 9852/10000 (99%)] [Best Epoch:21 Loss:0.040721 Acc:0.986300] [Best Step:8400 Loss:0.040721 Acc:0.986300]
[23/99] [Step:9200] [Train Loss:0.010249 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.039841 Acc:0.987700 9877/10000 (99%)] [Best Epoch:23 Loss:0.039841 Acc:0.987700] [Best Step:9200 Loss:0.039841 Acc:0.987700]
[23/99] [Step:9200] [Train Loss:0.058732 Acc:0.981993] [Val Loss:0.040197 Acc:0.987500 9875/10000 (99%)] [Best Epoch:23 Loss:0.039841 Acc:0.987700] [Best Step:9200 Loss:0.039841 Acc:0.987700] [3.40s 78.0s]
[24/99] [Step:9300] [Train Loss:0.195446 Acc:0.929688 12672/60000 (21%)] [Val Loss:0.048716 Acc:0.985300 9853/10000 (99%)] [Best Epoch:23 Loss:0.039841 Acc:0.987700] [Best Step:9200 Loss:0.039841 Acc:0.987700]
[24/99] [Step:9400] [Train Loss:0.097247 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.040279 Acc:0.987700 9877/10000 (99%)] [Best Epoch:23 Loss:0.039841 Acc:0.987700] [Best Step:9200 Loss:0.039841 Acc:0.987700]
[24/99] [Step:9500] [Train Loss:0.006016 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.038646 Acc:0.987700 9877/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[24/99] [Step:9600] [Train Loss:0.045985 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.042122 Acc:0.986400 9864/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[24/99] [Step:9600] [Train Loss:0.057982 Acc:0.981843] [Val Loss:0.042115 Acc:0.986500 9865/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700] [3.45s 81.5s]
[25/99] [Step:9700] [Train Loss:0.214298 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.044468 Acc:0.986300 9863/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[25/99] [Step:9800] [Train Loss:0.025256 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.039226 Acc:0.987800 9878/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[25/99] [Step:9900] [Train Loss:0.011093 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.039130 Acc:0.987400 9874/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[25/99] [Step:10000] [Train Loss:0.010913 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.039483 Acc:0.986900 9869/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[25/99] [Step:10000] [Train Loss:0.057631 Acc:0.982209] [Val Loss:0.040123 Acc:0.987200 9872/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700] [3.40s 84.9s]
[26/99] [Step:10100] [Train Loss:0.103644 Acc:0.945312 12672/60000 (21%)] [Val Loss:0.042485 Acc:0.986800 9868/10000 (99%)] [Best Epoch:24 Loss:0.038646 Acc:0.987700] [Best Step:9500 Loss:0.038646 Acc:0.987700]
[26/99] [Step:10200] [Train Loss:0.013324 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.038489 Acc:0.988800 9888/10000 (99%)] [Best Epoch:26 Loss:0.038489 Acc:0.988800] [Best Step:10200 Loss:0.038489 Acc:0.988800]
[26/99] [Step:10300] [Train Loss:0.012475 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.039427 Acc:0.986500 9865/10000 (99%)] [Best Epoch:26 Loss:0.038489 Acc:0.988800] [Best Step:10200 Loss:0.038489 Acc:0.988800]
[26/99] [Step:10400] [Train Loss:0.004220 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.061492 Acc:0.981100 9811/10000 (98%)] [Best Epoch:26 Loss:0.038489 Acc:0.988800] [Best Step:10200 Loss:0.038489 Acc:0.988800]
[26/99] [Step:10400] [Train Loss:0.056136 Acc:0.982326] [Val Loss:0.037145 Acc:0.988800 9888/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800] [3.33s 88.2s]
[27/99] [Step:10500] [Train Loss:0.072009 Acc:0.992188 12672/60000 (21%)] [Val Loss:0.045992 Acc:0.985900 9859/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[27/99] [Step:10600] [Train Loss:0.019295 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.040075 Acc:0.988400 9884/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[27/99] [Step:10700] [Train Loss:0.107911 Acc:0.968750 38272/60000 (64%)] [Val Loss:0.039755 Acc:0.988000 9880/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[27/99] [Step:10800] [Train Loss:0.022710 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.040754 Acc:0.987200 9872/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[27/99] [Step:10800] [Train Loss:0.054278 Acc:0.983042] [Val Loss:0.037527 Acc:0.988800 9888/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800] [3.40s 91.6s]
[28/99] [Step:10900] [Train Loss:0.103868 Acc:0.976562 12672/60000 (21%)] [Val Loss:0.040882 Acc:0.987400 9874/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[28/99] [Step:11000] [Train Loss:0.010534 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.040927 Acc:0.987400 9874/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[28/99] [Step:11100] [Train Loss:0.068444 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.040524 Acc:0.987200 9872/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[28/99] [Step:11200] [Train Loss:0.016629 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.039938 Acc:0.987200 9872/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[28/99] [Step:11200] [Train Loss:0.053462 Acc:0.983609] [Val Loss:0.040254 Acc:0.988300 9883/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800] [3.47s 95.1s]
[29/99] [Step:11300] [Train Loss:0.196940 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.047346 Acc:0.986000 9860/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[29/99] [Step:11400] [Train Loss:0.165260 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.039782 Acc:0.987600 9876/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[29/99] [Step:11500] [Train Loss:0.007553 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.038644 Acc:0.987900 9879/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[29/99] [Step:11600] [Train Loss:0.002496 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.037938 Acc:0.988900 9889/10000 (99%)] [Best Epoch:26 Loss:0.037145 Acc:0.988800] [Best Step:10400 Loss:0.037145 Acc:0.988800]
[29/99] [Step:11600] [Train Loss:0.052934 Acc:0.984025] [Val Loss:0.036203 Acc:0.988300 9883/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300] [3.44s 98.5s]
[30/99] [Step:11700] [Train Loss:0.047502 Acc:0.984375 12672/60000 (21%)] [Val Loss:0.043898 Acc:0.986900 9869/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[30/99] [Step:11800] [Train Loss:0.011684 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.037857 Acc:0.988700 9887/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[30/99] [Step:11900] [Train Loss:0.004318 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.040204 Acc:0.987700 9877/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[30/99] [Step:12000] [Train Loss:0.008329 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.036750 Acc:0.988100 9881/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[30/99] [Step:12000] [Train Loss:0.048677 Acc:0.984608] [Val Loss:0.038886 Acc:0.987900 9879/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300] [3.38s 101.9s]
[31/99] [Step:12100] [Train Loss:0.099696 Acc:0.976562 12672/60000 (21%)] [Val Loss:0.053687 Acc:0.984600 9846/10000 (98%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[31/99] [Step:12200] [Train Loss:0.032467 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.039723 Acc:0.987900 9879/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[31/99] [Step:12300] [Train Loss:0.040111 Acc:0.968750 38272/60000 (64%)] [Val Loss:0.042953 Acc:0.987700 9877/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[31/99] [Step:12400] [Train Loss:0.012863 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.039572 Acc:0.987900 9879/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[31/99] [Step:12400] [Train Loss:0.049349 Acc:0.984825] [Val Loss:0.038828 Acc:0.988100 9881/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300] [3.36s 105.3s]
[32/99] [Step:12500] [Train Loss:0.072450 Acc:0.984375 12672/60000 (21%)] [Val Loss:0.042399 Acc:0.986600 9866/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[32/99] [Step:12600] [Train Loss:0.074027 Acc:0.976562 25472/60000 (42%)] [Val Loss:0.037653 Acc:0.988200 9882/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[32/99] [Step:12700] [Train Loss:0.023814 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.039392 Acc:0.987800 9878/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[32/99] [Step:12800] [Train Loss:0.038161 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.037934 Acc:0.988100 9881/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[32/99] [Step:12800] [Train Loss:0.048948 Acc:0.984658] [Val Loss:0.040324 Acc:0.988400 9884/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300] [3.38s 108.7s]
[33/99] [Step:12900] [Train Loss:0.141661 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.048133 Acc:0.985400 9854/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[33/99] [Step:13000] [Train Loss:0.017906 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.036485 Acc:0.988100 9881/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[33/99] [Step:13100] [Train Loss:0.025618 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.036212 Acc:0.989200 9892/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[33/99] [Step:13200] [Train Loss:0.007274 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.036474 Acc:0.988600 9886/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[33/99] [Step:13200] [Train Loss:0.047984 Acc:0.985191] [Val Loss:0.036490 Acc:0.988300 9883/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300] [3.36s 112.0s]
[34/99] [Step:13300] [Train Loss:0.107023 Acc:0.968750 12672/60000 (21%)] [Val Loss:0.040717 Acc:0.987900 9879/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[34/99] [Step:13400] [Train Loss:0.011424 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.038674 Acc:0.988300 9883/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[34/99] [Step:13500] [Train Loss:0.005914 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.040367 Acc:0.988200 9882/10000 (99%)] [Best Epoch:29 Loss:0.036203 Acc:0.988300] [Best Step:11600 Loss:0.036203 Acc:0.988300]
[34/99] [Step:13600] [Train Loss:0.014052 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.035886 Acc:0.989000 9890/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[34/99] [Step:13600] [Train Loss:0.044536 Acc:0.985774] [Val Loss:0.036242 Acc:0.988600 9886/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000] [3.36s 115.4s]
[35/99] [Step:13700] [Train Loss:0.139209 Acc:0.976562 12672/60000 (21%)] [Val Loss:0.041450 Acc:0.987300 9873/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[35/99] [Step:13800] [Train Loss:0.009460 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.037457 Acc:0.987700 9877/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[35/99] [Step:13900] [Train Loss:0.039068 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.037525 Acc:0.987500 9875/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[35/99] [Step:14000] [Train Loss:0.049194 Acc:0.984375 51072/60000 (85%)] [Val Loss:0.037311 Acc:0.988400 9884/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[35/99] [Step:14000] [Train Loss:0.046671 Acc:0.985624] [Val Loss:0.036848 Acc:0.987900 9879/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000] [3.38s 118.8s]
[36/99] [Step:14100] [Train Loss:0.186945 Acc:0.921875 12672/60000 (21%)] [Val Loss:0.044100 Acc:0.986900 9869/10000 (99%)] [Best Epoch:34 Loss:0.035886 Acc:0.989000] [Best Step:13600 Loss:0.035886 Acc:0.989000]
[36/99] [Step:14200] [Train Loss:0.008808 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.035348 Acc:0.988700 9887/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[36/99] [Step:14300] [Train Loss:0.026628 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.036591 Acc:0.988900 9889/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[36/99] [Step:14400] [Train Loss:0.014738 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.036654 Acc:0.988600 9886/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[36/99] [Step:14400] [Train Loss:0.044300 Acc:0.985974] [Val Loss:0.035693 Acc:0.989300 9893/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700] [3.36s 122.1s]
[37/99] [Step:14500] [Train Loss:0.143438 Acc:0.953125 12672/60000 (21%)] [Val Loss:0.044170 Acc:0.986200 9862/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[37/99] [Step:14600] [Train Loss:0.011590 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.041603 Acc:0.987500 9875/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[37/99] [Step:14700] [Train Loss:0.012192 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.042507 Acc:0.988100 9881/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[37/99] [Step:14800] [Train Loss:0.018301 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.036642 Acc:0.989000 9890/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[37/99] [Step:14800] [Train Loss:0.044769 Acc:0.985674] [Val Loss:0.035822 Acc:0.989200 9892/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700] [3.42s 125.5s]
[38/99] [Step:14900] [Train Loss:0.199537 Acc:0.929688 12672/60000 (21%)] [Val Loss:0.040664 Acc:0.987400 9874/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[38/99] [Step:15000] [Train Loss:0.007668 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.036909 Acc:0.988200 9882/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[38/99] [Step:15100] [Train Loss:0.014516 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.038753 Acc:0.988800 9888/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[38/99] [Step:15200] [Train Loss:0.005924 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.037083 Acc:0.988600 9886/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[38/99] [Step:15200] [Train Loss:0.044418 Acc:0.986057] [Val Loss:0.038206 Acc:0.987700 9877/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700] [3.43s 129.0s]
[39/99] [Step:15300] [Train Loss:0.135682 Acc:0.937500 12672/60000 (21%)] [Val Loss:0.041830 Acc:0.987500 9875/10000 (99%)] [Best Epoch:36 Loss:0.035348 Acc:0.988700] [Best Step:14200 Loss:0.035348 Acc:0.988700]
[39/99] [Step:15400] [Train Loss:0.005654 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.035164 Acc:0.989200 9892/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[39/99] [Step:15500] [Train Loss:0.056215 Acc:0.992188 38272/60000 (64%)] [Val Loss:0.038227 Acc:0.988500 9885/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[39/99] [Step:15600] [Train Loss:0.021542 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.038088 Acc:0.988300 9883/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[39/99] [Step:15600] [Train Loss:0.042998 Acc:0.986241] [Val Loss:0.037805 Acc:0.989600 9896/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [3.46s 132.4s]
[40/99] [Step:15700] [Train Loss:0.358305 Acc:0.906250 12672/60000 (21%)] [Val Loss:0.043509 Acc:0.987400 9874/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[40/99] [Step:15800] [Train Loss:0.004372 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.037114 Acc:0.988600 9886/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[40/99] [Step:15900] [Train Loss:0.034244 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.037929 Acc:0.988800 9888/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[40/99] [Step:16000] [Train Loss:0.017101 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.038995 Acc:0.987400 9874/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[40/99] [Step:16000] [Train Loss:0.042670 Acc:0.986807] [Val Loss:0.041394 Acc:0.988200 9882/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [3.32s 135.7s]
[41/99] [Step:16100] [Train Loss:0.140571 Acc:0.968750 12672/60000 (21%)] [Val Loss:0.046293 Acc:0.986400 9864/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[41/99] [Step:16200] [Train Loss:0.057397 Acc:0.992188 25472/60000 (42%)] [Val Loss:0.050689 Acc:0.985300 9853/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[41/99] [Step:16300] [Train Loss:0.001448 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.037245 Acc:0.988900 9889/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[41/99] [Step:16400] [Train Loss:0.049858 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.037049 Acc:0.988900 9889/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[41/99] [Step:16400] [Train Loss:0.042055 Acc:0.986740] [Val Loss:0.038260 Acc:0.988500 9885/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [3.30s 139.0s]
[42/99] [Step:16500] [Train Loss:0.099408 Acc:0.976562 12672/60000 (21%)] [Val Loss:0.039224 Acc:0.988200 9882/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[42/99] [Step:16600] [Train Loss:0.006981 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.036526 Acc:0.989000 9890/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[42/99] [Step:16700] [Train Loss:0.001718 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.038471 Acc:0.988100 9881/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[42/99] [Step:16800] [Train Loss:0.037410 Acc:0.992188 51072/60000 (85%)] [Val Loss:0.036508 Acc:0.988500 9885/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[42/99] [Step:16800] [Train Loss:0.041504 Acc:0.986840] [Val Loss:0.035767 Acc:0.989100 9891/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [3.39s 142.4s]
[43/99] [Step:16900] [Train Loss:0.083820 Acc:0.976562 12672/60000 (21%)] [Val Loss:0.041104 Acc:0.987500 9875/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[43/99] [Step:17000] [Train Loss:0.024647 Acc:0.984375 25472/60000 (42%)] [Val Loss:0.037562 Acc:0.987900 9879/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[43/99] [Step:17100] [Train Loss:0.072865 Acc:0.984375 38272/60000 (64%)] [Val Loss:0.036915 Acc:0.988800 9888/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[43/99] [Step:17200] [Train Loss:0.003012 Acc:1.000000 51072/60000 (85%)] [Val Loss:0.035968 Acc:0.988600 9886/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[43/99] [Step:17200] [Train Loss:0.050400 Acc:0.984358] [Val Loss:0.037091 Acc:0.989100 9891/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [3.45s 145.9s]
[44/99] [Step:17300] [Train Loss:0.178450 Acc:0.945312 12672/60000 (21%)] [Val Loss:0.055685 Acc:0.983600 9836/10000 (98%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[44/99] [Step:17400] [Train Loss:0.009561 Acc:1.000000 25472/60000 (42%)] [Val Loss:0.036629 Acc:0.988800 9888/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
[44/99] [Step:17500] [Train Loss:0.007095 Acc:1.000000 38272/60000 (64%)] [Val Loss:0.035958 Acc:0.988700 9887/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200]
Early Stop With step: 17500
[44/99] [Step:17500] [Train Loss:0.051203 Acc:0.980807] [Val Loss:0.035958 Acc:0.988700 9887/10000 (99%)] [Best Epoch:39 Loss:0.035164 Acc:0.989200] [Best Step:15400 Loss:0.035164 Acc:0.989200] [2.47s 148.4s]
Test set: Average loss: 0.0352, Accuracy: 9892/10000 (99%)
Test set: Average loss: 0.0352, Accuracy: 9892/10000 (99%)
```

