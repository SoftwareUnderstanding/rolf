# ResNeSt-Pytorch
Implementation of novel backbone to do image classification for LCD and PCB  

## Model Installation:  
Please refer to the following webstite:  
https://github.com/zhanghang1989/ResNeSt  


## Run the model:  

1. command: python train_resneSt_gen.py -t  

    => For PCB data, there are lots of images estimated about 200k ea,  
    so we must divide a picke file into several pickles.  


2. command: python train_resneSt.py -t  
    => If the dataset is pretty smaller, so there is no need to divide a pickle file.(less than 10k ea)  


## Reference:  
The paper can be sourced to  
https://arxiv.org/abs/2004.08955  

