# SRGAN

Class project for ECE228 Machine Learning

Group Member: Xupeng Yu, Hongquan Zhang, Guoren Zhong

- This is our own implement of SRGAN baseed on Tensorflow2 and Tensorlayer.

## Usage:
- Training:
  Each of the three models has its own training code. To train each model, just run:
> python train_srcnn.py <br/>
> python train_srresnet.py <br/>
> python train_srgan.py <br/>
- Testing:
  The three models have the same evaluation process. To test each model,run:
> python evaluate.py -m MODEL #The model youwant to test,here we support ['srcnn','srresnet','srgan']
  
## Reference:
- [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network] [https://arxiv.org/abs/1609.04802]
- [2] [Image Super-Resolution Using Deep Convolutional Networks] [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html] 
