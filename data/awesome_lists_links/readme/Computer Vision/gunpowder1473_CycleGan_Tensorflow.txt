# Style Transfer based on GAN network

This is a python implementation of cycleGan based on TensorFlow，which is used to realized style transfer.

Original paper: https://arxiv.org/abs/1703.10593

## Requirement

- TensorFlow 1.12

- matplotlib

- py-opencv 3.42

- numpy


## Training

To get started with training you can run
```
python GANTrain.py
```
Notice that I used `tf.app.flags` to control the options shown below, you need to edit them in `GANTrain.py`:

 - `imgA` : the location of the image A saved, image A is one of two input images of cycleGan.

 - `imgB` : the location of the image B saved, image B is one of two input images of cycleGan.

 - `imgC` : the location of the validate image saved, I now using TensorBoard and the relevant codes have been commented out.This option is abandoned for now.

 - `val_out` : the location of result of validate image saved.Abandoned too.

 - `checkpoint` : the location of ckpt model and TensorBoard summary saved.`../checkout` by default.

 - `Norm` : the norm method to use.Could be `BATCH` or `INSTANCE`.`INSTANCE` by default.

 - `learning_rate` : the initial learning rate. `2e-4` by default.

 - `start_step` : the start step if using `linear_decay`. `100000` by default, which means learning rate remains unchanged during first 10000 steps, the start to reduce linearly.

 - `end_step` : the end step if using `linear_decay`. `200000` by default, which means learning rate should be 0 after 200000 steps.

 - `max_to_keep` : the number of saved model kept on the same time. `10` by default.

- `summary_iter` : the interval training steps of every summary step. `10` by default, which means summary every 10 steps.

- `save_iter` : the interval training steps of every save step. `200` by default, which means save model every 200 steps.

- `val_iter` : the interval training steps of every validate step. Abandoned.

- `batch_size` : the batch size of training. `1` by default. This parameter depend on GPU memory.

- `lambda1` : the weight of cycleLoss and identifyLoss. `10` by default.

- `lambda2` : the weight of cycleLoss and identifyLoss. `10` by default.

- `ngf`: the number of filters in first convolution layer of Generator. `64` by default.

- `img_size`: the input size of the GAN. `256` by default.

- `USE_E`: to choose wheter use the original method or improved method. `False` by default.


## Testing

To get started with testing you can run
```
python GANTest.py
```
Notice that I used `tf.app.flags` to control the options shown below, you need to edit them in `GANTest.py`:

 - `input` : the location of the input image saved.

 - `output` : the location of the result saved.

 - `checkpoint` : the location of ckpt model.`../checkout` by default.

 - `Norm` : the norm method to use.Could be `BATCH` or `INSTANCE`.`INSTANCE` by default.

 - `status` : change to transfer from X to Y or from Y to X. Could be `X2Y` or `Y2X`.

- `batch_size` : the batch size of testing. `1` by default. This parameter depend on GPU memory.

- `ngf`: the number of filters in first convolution layer of Generator. `64` by default.

- `img_size`: the input size of the GAN. `256` by default.

- `USE_E`: to choose wheter use the original method or improved method. `False` by default.

## Note

1. Do not use `BATCH` if you do not have enough GPU memory to support a big batch, `INSTANCE` performs much better when set `batch_size` to 1.

2. `USE_E` is a improvement. I found it works in style transfer for summer2winter, which is the paper's author's data set, see comparison in next sector. But have not proved on other data set.

3. By default, every training step cost 0.3s on GTX1080Ti.

4. Author's data set can be found in https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

## Result

Results for transfer between SD dolls and Anime characters.

<img src="./result/SD2Anime.png" style="zoom:50%">

<img src="./result/Anime2SD.png" style="zoom:50%">

Results for transfer between zebra and horse.

<img src="./result/horsezebra.png" style="zoom:50%">

Results for summer to winnter, comparing original method and improved method.

<img src="./result/summerwinter.png" style="zoom:50%">

## Problem
In the experiment, I found this problem without using identifyLoss. Also for using resizeConv2D(http://distill.pub/2016/deconv-checkerboard/) rather than transposeConv2D.

<img src="./result/question.png" style="zoom:50%">

And in my implement of BigGan, I got following similar problem when change relu to abs(that is a miss) in loss.

<img src="./result/question2.png" style="zoom:50%">

Contact me if you have any idea.
