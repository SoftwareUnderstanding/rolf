# Deep Convolution Generative Adversarial Networks

Implementation  on pytorch.

code based on https://github.com/soumith/dcgan.torch and https://github.com/pytorch/examples/tree/master/dcgan

original article: https://arxiv.org/abs/1511.06434

used datasets: imagenet(32x32), lsun(conference_room), food-101

required nvidia graphic card

### Usage
```
as default behaviour - training on lsun dataset
usage: main.py [--dataset DATASET] [--dataroot DATAROOT] [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
               [--beta1 BETA1] [--netG NETG] [--netD NETD]
               [--outf OUTPUTFOLD] [--manualSeed SEED] [--train_svm]

arguments:
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTPUTFOLD     folder to output images and model checkpoints
  --manualSeed SEED     manual seed
  --train_svm           enable train svm using saved features
  
```

* *main.py* - train dcgan

* *arithmetic.py* - create some new images applying "image arithmetic"

* *classifier_svm.py* - without flag --train_svm just create and store features, with - download features and train svm. Metrics: accuracy for the whole dataset, precision and recall for each class.

* *extract_imagenet.py* - create pictures from pickle

* *get_samples.py* - use pretrained generator to get samples from noise


![alt text](https://github.com/Annusha/dcgan/blob/master/images/means.png)

![alt text](https://github.com/Annusha/dcgan/blob/master/images/A_B.png)
