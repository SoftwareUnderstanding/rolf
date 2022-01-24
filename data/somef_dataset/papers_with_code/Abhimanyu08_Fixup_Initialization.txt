# Fixup_Intialisation
Initialize your ResNets with any default inits you want + Fixup init from the paper ["Fixup Initialization: Residual Learning without Normalization".](https://arxiv.org/abs/1901.09321)

Fixup Initialization was introduced by Zhang et.al in the paper "Fixup Initialization: Residual Learning without Normalization". This intialization aims at removing the need for BatchNorm in the network and provide similar results as network with one. This init can be potentially helpful in runnning entire training process in fp16 precision since the BatchNorm requires computation in fp32. This can lead to reduction in training time.
To initialize the model using Fixup just type model = resnetX(init_func = 'Fixup'). To use any other init for.eg Kaiming init, you need to use argument names from the pytorch library's nn.init module i.e model = resnetX(init_func = 'kaiming_normal_'). X here denotes the no of Convolutional layers in you resnet for eg. resnet50.
