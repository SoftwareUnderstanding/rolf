# Aerial Imaging Classification with a Lightweight Network

Brian Chan   

### Abstract: 

Edge computing and satellite imagery trends demand lightweight algorithms for computational analysis. In this paper, I investigate applying SqueezeNet, a neural network with AlexNet-level accuracy and 510x less size, to the task of aerial imaging land use classification with the NWPU-RESISC45 dataset. By the end of the paper, we find that the models developed using transfer learning are not as robust as heftier models and invite further investigation to maintain their performance on aerial imaging.

Original SqueezeNet paper:
https://arxiv.org/abs/1602.07360

### Summary of files:

**sqznet_models.py**: contains two functions that create the SqueezeNet networks: the first being one without the top layers, and the second being one for fine-tuning, as it can be adapted to create SqueezeNet with any desired following layers to allow SqueezeNet's parameters to be adjusted.

**sqznet_transfer.py**: runs the bottom layers of a SqueezeNet to generate data for transfer learning. Then contains functions that can be utilized to train "top model" neural networks that receive the aforementioned output as their input.

**sqznet_predict.py**: is utilized to load given models and weights, and then used to make predictions as well as output metrics and a confusion matrix.

**sqznet_finetune.py**: is used as driver code to create and then train a tunable squeezenet + FC layers from code_squeezenet.py.

**vgg_transfer.py**: allows basic implementation of transfer learning with a vgg-16 model: creates an instance without top levels, captures its outputs, and then uses those outputs as input to another trainable model.

----
### Misc Files

**move.py**: a script that can be adapted to move image samples between test/train/validation directories.

**predict_CNN.py + train_CNN.py**: Basic CNN image classifier used very early on during this project (unrelated to final CS230 report)


