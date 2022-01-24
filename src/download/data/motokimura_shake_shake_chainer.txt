# Shake-Shake Regularization
A Chainer implementation of [Shake-Shake regularization](https://arxiv.org/abs/1705.07485).

## Results on CIFAR-10 (Shake-ResNet)

| Model                         | Test Error (this implementation) | Test Error (in paper)    |
|:------------------------------|:--------------------------------:|:------------------------:|
| Shake-ResNet-26 2x32d (S-S-I) | Not tested yet                   | 3.55 (average of 3 runs) |
| Shake-ResNet-26 2x64d (S-S-I) | **2.85 (1 run)**                 | 2.98 (average of 3 runs) |
| Shake-ResNet-26 2x96d (S-S-I) | Not tested yet                   | 2.86 (average of 5 runs) |

![](contents/resnet_err.png)

* The model Shake-ResNet-26 2x64d (S-S-I) is trained with batch size 128, and initial learning rate 0.1.

## Results on CIFAR-100 (Shake-ResNeXt)

*Currently Shake-ResNeXt is not implemented.. Your contribution is more than welcome!*

## Dependency

* [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) 

## Usage

### 0. Clone this repo

```
$ PROJ_DIR=~/shake_shake_chainer  # assuming you clone this repo to your home directory

$ git clone https://github.com/motokimura/shake_shake_chainer.git $PROJ_DIR
```

### 1. Build Docker image

Build docker image to setup the environment to train/test the model with Shake-Shake regularization. 

```
$ cd $PROJ_DIR/docker
$ bash build.sh
```

### 2. Train Shake-ResNet model with CIFAR-10

Run docker container by following:

```
$ cd $PROJ_DIR/docker
$ bash run.sh
```

Now you should be inside the docker container you ran. Start training by following:

```
$(docker) cd /workspace
$(docker) python train_model.py
```

You can check training status and test accuracy from TensorBoard:

```
# Open another terminal window outside the container and type:
$ cd $PROJ_DIR/docker
$ bash exec.sh

# Now you should be inside the container already running. Start TensorBoard by following:
$(docker) tensorboard --logdir /workspace/results
```

Then, open `http://localhost:6006` from your browser.

## References

### Papers

* Gastaldi, Xavier. "Shake-Shake regularization." [[arXiv]](https://arxiv.org/abs/1705.07485)
* Gastaldi, Xavier. "Shake-Shake regularization of 3-branch residual networks." [[ICLR2017 Workshop]](https://openreview.net/forum?id=HkO-PCmYl)

### Other implementations

* nutszebra's Chainer implementation. [[GitHub]](https://github.com/nutszebra/shake_shake)
* akitotakeki's Chainer implementation. [[Gist]](https://gist.github.com/akitotakeki/c82a3bb38c930cd295628cfa1e29fdd7)
* owruby's Pytorch implementation. [[GitHub]](https://github.com/owruby/shake-shake_pytorch)
* Author's Lua implementation. [[GitHub]](https://github.com/xgastaldi/shake-shake)

## License

[MIT License](LICENSE)
