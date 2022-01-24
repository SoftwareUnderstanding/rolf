# keras_DCGAN_transfer_learning
Building a Keras DCGAN using a pre-trained Torch generator

Trying to use the power of Deep Learning to generate new images, but not willing to invest in hindend GPU? Transfer learning could be a solution.

`keras_dcgan.py` builds and trains a DCGAN model (https://arxiv.org/abs/1511.06434). You can choose to either use the generator from the original paper, or a similar FloyedHub implementation (https://github.com/ReDeiPirati/dcgan).

In the latter case, you can choose to copy the weights of the pre-trained model. Those you can download from here https://www.floydhub.com/redeipirati/datasets/dcgan-300-epochs-models/1/netG_epoch_299.pth, but for convenience I have already include the model in the `weights` directory.

Note that the FloyHub model is built with Torch (see `torch_dcgan.py`). To translate this to Keras `Torch2Keras.py`:

1. builts the Torch model and loads its weights 
2. builts a Keras model with the same architecture
3. extracts weights as numpy arrays from the first model and feeds them to the second
4. saves weights of Keras model in `weights` sub-directory

The converter works only for the given architecture, for a more general one try https://github.com/nerox8664/pytorch2keras

The FloyHub pre-trained model is trained on the `Labeled Faces in the Wild' database (http://vis-www.cs.umass.edu/lfw/). The corresponding Keras implementation gives:

![FloyHub Generated Images](/generated_imgs/res_1.png)

To apply this to your project see `main.py`. I am currentely working on switching from 'faces in the wild' to 'simpsons faces'.
