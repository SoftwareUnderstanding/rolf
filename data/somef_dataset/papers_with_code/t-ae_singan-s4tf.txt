# SinGAN on Swift for TensorFlow

- arXiv: [SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164)
- [Supplementary Material](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shaham_SinGAN_Learning_a_ICCV_2019_supplemental.pdf)
- Offiicial implementation: [tamarott/SinGAN](https://github.com/tamarott/SinGAN)

## Run

[Swift for TensorFlow](https://github.com/tensorflow/swift) and some python dependencies are required.  
It's not trainable with latest release v0.6.0 due to its AD bug. Use development snapshot.

I recommend this Dockerfile:  
https://github.com/t-ae/s4tf-docker

### Commands

```bash
$ swift run -c release SinGAN Input/ballons.png
```

TensorBoard log will be generated under `logdir`.

```bash
$ tensorboard --logdir logdir/
```

## Example

### Super resolution

|  Original  |  SR  |
| ---- | ---- |
|  ![33039_LR](https://user-images.githubusercontent.com/12446914/72676461-d9d03480-3ad4-11ea-8fd0-55beb75ddde9.png)  |  ![super_resolution5](https://user-images.githubusercontent.com/12446914/72676479-06844c00-3ad5-11ea-9845-f1d864837e1c.png)  |

### Multiple sizes

![multisize_181x181](https://user-images.githubusercontent.com/12446914/72676495-29aefb80-3ad5-11ea-9f16-e90c673a3a6b.png)
![multisize_181x369](https://user-images.githubusercontent.com/12446914/72676496-29aefb80-3ad5-11ea-8dfd-bab322a940f0.png)
![multisize_293x181](https://user-images.githubusercontent.com/12446914/72676497-2a479200-3ad5-11ea-980d-8b12b6cd40c3.png)
![multisize_592x181](https://user-images.githubusercontent.com/12446914/72676498-2a479200-3ad5-11ea-81f0-5d2d0a21881c.png)

More examples in [Results directory](https://github.com/t-ae/singan-s4tf/tree/master/Results).

## Differences from original

### Instance norm instead of batch norm

Original implementation uses batch norm. I afraid it's problematic.  
SinGAN is trained with single image. It means batch size is always 1.  
Therefore batch norm works like instance norm while training.  
But when it comes to inference phase, batch norm uses running stats of training phase. It can be much different from training phase.  

To avoid this, I simply replaced batch norm with instance norm.


### Cease WGAN-GP training

As I wrote in [the issue](https://github.com/tamarott/SinGAN/issues/59), original implementation of gradient penalty looks wrong.  
Anyway S4TF doesn't support higher-order differentiaion for now. So I decided not to use WGAN-GP.

### Use spectral normalization

Since I didn't use WGAN-GP, I need other techniques to stabilize training.  
I employed [spectral normalization](https://arxiv.org/abs/1802.05957) and use hinge loss.
