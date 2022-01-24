# SSD

Implentation of https://arxiv.org/pdf/1512.02325.pdf

Dependencies:

* pytorch(0.4, might work with 1.0)
* lxml
* visdom
* python 3.6

## Setup

Pull the repo and then run: git submodule update --init --recursive

## Training

In order to run training you will need the VOC (2007 or 2012) dataset seperated into images and annotaions.

These paths need to be supplied to the dataloader in the file ssd_train.py

You can then run training by first starting visdom in terminal by running the command: visdom

Then run: python ssd_train.py

If you open the browers and point it to where visdom is running you should see something like this:

![alt text](samples/viz/visdom_training_example.png)

The main thing to note is the repo doesn't have the augmentation code or the different backbones I've been experimenting with yet.

## Testing

python ssd_eval.py runs the model over all of the images in samples/test_images with nms. Here are a couple examples, I forgot to add the classifications:

![alt text](samples/pred_images/pred_0.png)
![alt text](samples/pred_images/pred_1.png)
![alt text](samples/pred_images/pred_2.png)
![alt text](samples/pred_images/pred_3.png)

The model runs at over 50 FPS on a 1080ti.