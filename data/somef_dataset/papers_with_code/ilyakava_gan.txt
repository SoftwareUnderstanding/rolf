# cGANs with Multi-Hinge Loss: tensorflow TPU and GPU implementation

![MHGAN sample](./tensorflow_gan/examples/self_attention_estimator/images/both_rows.jpg)

This code is forked from TF-GAN. TF-GAN is a lightweight library for training and evaluating [Generative
Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661).

This code implements cGANs with Multi-Hinge Loss from [this paper](https://arxiv.org/abs/1912.04216), for fully and semi supervised settings.
It uses the Imagenet, Cifar100, Cifar10 datasets.

Please cite:

```
@InProceedings{Kavalerov_2021_WACV,
author = {Kavalerov, Ilya and Czaja, Wojciech and Chellappa, Rama},
title = {A Multi-Class Hinge Loss for Conditional GANs},
booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
month = {January},
year = {2021}
}
```


## What's in this repository

- Baseline SAGAN
- Working Baseline ACGAN with many auxiliary loss choices
- Multi-Hinge GAN
- batched intra-fid calculation for significant speedup
    + Imagenet 1000 class intra-fid at 2.5k images per class takes < 18h on v3-8 TPU
- K+1 GANs
- print out eval metrics in google cloud without tensorboard and with no more than 6GB of mem required.

These work on both TPU and GPU, but the TPU implementation has more features.

This code builds off of the tf example for [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).
See more info [here](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/examples/self_attention_estimator).
All other examples are ignored.
Because of the design choices everything outside of the SAGAN example, including all tests not specified below, may no longer work.

## Download pretrained models

[Downloads root is here](https://drive.google.com/drive/folders/1SZeCaPrEqRYXUyhWi_BLEEpiJFdtpX1B?usp=sharing)

Note: in some cases multiple checkpoints of weights are given. To control which weights are loaded edit the number in txt 'checkpoint' file in each model directory.

Note: to run eval you need the original dataset.

- Imagenet 128x128 SAGAN
    - [download pretrained baseline SAGAN model here](https://drive.google.com/drive/folders/1oLBISsbAbs5G-emQ6Roix2-LOQiZEJy1?usp=sharing)
    - step 1130000 has FID 21.9846 and IS 52.8251
    - used in `/gan/tensorflow_gan/examples/gpu/example_genimages_imagenet128_baseline.sh`
    - used in `/gan/tensorflow_gan/examples/gpu/example_eval_imagenet128_baseline.sh`
- Imagenet 128x128 MHGAN
    - [download pretrained MHGAN model here](https://drive.google.com/drive/folders/1acQL6rhSIuIzpPvymclbw6cNaee7N1Fy?usp=sharing)
    - has FID 19.1119 and IS 60.9724
    - used in `/gan/tensorflow_gan/examples/gpu/example_genimages_imagenet128_expt.sh`
    - used in `/gan/tensorflow_gan/examples/gpu/example_eval_imagenet128_expt.sh`
- Imagenet 64x64 high-fidelity-low-diversity contrast
    - contrast the normal SAGAN model with the high-fidelity-low-diversity one at the later checkpoint
    - [download pretrained models here](https://drive.google.com/drive/folders/1GfuPWCks08v1Ftgdd7jHL-_TRs7QxgZ-?usp=sharing)
    - step 999999 has FID 15.482412 and IS 19.486443
    - step 1014999 has FID 10.249778 and IS 29.656748 but very low diversity
    - used in `/gan/tensorflow_gan/examples/gpu/example_lowdiversity_eval_imagenet64.sh`
    - used in `/gan/tensorflow_gan/examples/gpu/example_lowdiversity_genimages_imagenet64.sh`
- Imagenet 128x128 high-fidelity-low-diversity contrast
    - contrast the normal SAGAN model with the high-fidelity-low-diversity one at the later checkpoint
    - [download pretrained models here](https://drive.google.com/drive/folders/1CGjsCqhRinxB3qRf0fmpAA8fYxauOiO6?usp=sharing)
    - step 580000 has IS 47.79 and FID 17.10
    - step 585000 has IS 169.68 and FID 8.87 but very low diversity    
    - used in `/gan/tensorflow_gan/examples/gpu/example_lowdiversity_genimages_imagenet128.sh`
    - 64000 images (class is random) sampled from each of these models are [available here as tarball](https://drive.google.com/drive/folders/1OKMHYQSZCNHQBsmNCmU3xLFx-Y42NfYC?usp=sharing)
    - [Browser viewable images here](https://drive.google.com/drive/folders/1iN7io65N7QzkiXYx079SRWr4naOCHbHU?usp=sharing) of 36 images per class from each of these models.

## Performance on Imagenet128

- Baseline SAGAN runs the same as seen [here](https://github.com/tensorflow/gan/tree/master/tensorflow_gan/examples/self_attention_estimator), (best ever) IS 52.79 and FID 16.39 after 1M iters. Explodes soon after.
- MHGAN (best ever) IS 61.98 and FID 13.27 within 1M iter. Explodes around the same time.
- ACGAN with cross entropy does (best ever) IS 48.94 and FID 24.72.

Batch size of 1024, 1 D step per G step, 64 chan, and more as seen in `gan/tensorflow_gan/examples/tpu/imagenet128_baseline.sh`.
1M steps takes about 10 days on a v3-8 TPU.

## How to run

See scripts in:
- `gan/tensorflow_gan/examples/gpu/*.sh`
- `gan/tensorflow_gan/examples/tpu/*.sh`

To replicate the baseline:

- `gan/tensorflow_gan/examples/tpu/imagenet128_baseline.sh`

To replicate MHingeGAN:

- `gan/tensorflow_gan/examples/tpu/imagenet128.sh`

To eval:

- `gan/tensorflow_gan/examples/tpu/eval_ifid_imagenet128.sh`
- `gan/tensorflow_gan/examples/gpu/eval_imagenet128.sh`
- `gan/tensorflow_gan/examples/tpu/genimages_imagenet128.sh`

To run a small experiment, see for example:

- `/gan/tensorflow_gan/examples/gpu/cifar_ramawks69.sh`
- `/gan/tensorflow_gan/examples/tpu/cifar100.sh`

Such an experiment takes only an hour to get to 200k on a v3-8 TPU.

## Installation GPU

Use miniconda3, python 3.7.7, tensorflow 2.1, cuda/10.1.243, cudnn/v7.6.5. Install `venvtf2p1.yml`.

In general:


```
pip install tensorflow_datasets
pip install tensorflow_gan
pip uninstall -y tensorflow_probability
pip install tensorflow_probability==0.7
pip install pillow
```

## Setup GPU

```
export TFGAN_REPO=`pwd`
export PYTHONPATH=${TFGAN_REPO}/tensorflow_gan/examples:${PYTHONPATH}
export PYTHONPATH=${TFGAN_REPO}:${PYTHONPATH}
cd tensorflow_gan/examples
```

Then run a script in `gpu/`.

## Requesitioning TPU

See [Quickstart](https://cloud.google.com/tpu/docs/quickstart) but in general:

```
export PROJECT_NAME=xxx-xxx-xxx
export PROJECT_ID=${PROJECT_NAME}
gcloud config set project ${PROJECT_NAME}

export ZONE="europe-west4-a"
export REGION="europe-west4"
ctpu up --zone=${ZONE} --tpu-size="v3-8" --tf-version=2.1 --name=${TPU_NAME} --machine-type=n1-standard-1
```

For lower costs the following custom size is sufficient for a v3-8 TPU.
The following command uses an image that can be created after launching in the general way above.

```
gcloud beta compute instances create tpu-eu-1 --zone=${ZONE} --source-machine-image tf2p1 --custom-cpu 1 --custom-memory 6 --custom-vm-type n1
```

## (Re)Connect to TPU

In general:

```
gcloud compute ssh ${TPU_NAME} --zone=${ZONE}
```

OR if using a custom instance:

```
gcloud beta compute ssh --zone=${ZONE} ${TPU_NAME} --project=${PROJECT_NAME}
```

Once inside the TPU:

```
export ZONE="europe-west4-a"
export REGION="europe-west4"
export BUCKET_NAME="mybucket"
```

```
export PROJECT_NAME=xxx-xxx-xxx
export PROJECT_ID=${PROJECT_NAME}
export STORAGE_BUCKET=gs://${BUCKET_NAME}
export TPU_ZONE=${ZONE}
```

## Installation TPU

Use python3.7, tensorflow 2.1.

After launching run:

```
pip3.7 install --upgrade tensorflow_datasets --user
git clone --single-branch --branch dev https://github.com/ilyakavagan.git
pip3.7 install tensorflow_gan --user
```

## Setup TPU

```
export TFGAN_REPO=/home/ilyak/gan
export PYTHONPATH=${TFGAN_REPO}/tensorflow_gan/examples:${PYTHONPATH}
export PYTHONPATH=${TFGAN_REPO}:${PYTHONPATH}

cd gan/tensorflow_gan/examples

git pull origin dev
source tpu/retry.sh
```

## TPU monitoring

### Print eval metrics in the cloud

This requires a separate cpu instance, 6GB is enough memory.

Use `gan/tensorflow_gan/examples/print_tf_log.py`

### Monitoring in the cloud

Cloud tpu tensorboard crashes very often, but it does work during the 1st hour of training while there are few logfiles.
Run it with:

```
export TPU_IP=XX.XXX.X.X
tensorboard --logdir=${STORAGE_BUCKET}/experiments/${EXPERIMENT_NAME} --master_tpu_unsecure_channel=${TPU_IP}
```

Most useful is the [TPU profiling](https://cloud.google.com/tpu/docs/tensorboard-setup#static-trace-viewer):

```
capture_tpu_profile --port=8080 --tpu=${TPU_NAME} --tpu_zone=${ZONE} --logdir=gs://${BUCKET_NAME}/experiments/${EXPERIMENT_NAME}/logdir
```

### Downloading log files and monitoring locally (Not recommended)

[Install gsutil](https://cloud.google.com/storage/docs/gsutil_install)

`gsutil cp -R ${STORAGE_BUCKET}/experiments/${EXPERIMENT_NAME}/eval_eval ./${EXPERIMENT_NAME}/`

and launch tensorboard locally. Will incure high egress costs.

## TPU/Cloud Teardown

Check status with:

`ctpu status --zone=$ZONE --name=${TPU_NAME}`

Use the google cloud console. Note that TPUs and their CPU hosts have to be killed separately.
- https://console.cloud.google.com/compute/tpus
- https://console.cloud.google.com/compute/instances

Remove buckets with:

`gsutil rm -r gs://${BUCKET_NAME}/experiments/${EXPERIMENT_NAME}`

## Setup: Prepare Data

Open a python REPL and:

```
# if you don't do this, it will crash on Imagenet
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import tensorflow_datasets as tfds
```

Then depending on your dataset:

- `ds = tfds.load('imagenet_resized/64x64', split="train", data_dir="/mydir/data/tfdf")`
- `ds = tfds.load('cifar100', split="train", data_dir="/mydir/data/tfdf")`
- `ds = tfds.load("cifar10", split="train", data_dir="/mydir/data/tfdf")`

For Imagenet at 128x128, you need to manually download the data and place the compressed files in the tfdf data_dir downloads folder before running 

`ds = tfds.load("imagenet2012", split="train", data_dir="gs://mybucket/data")`

Be forewarned Imagenet128 takes several hours to download and over 24 hours to setup initially on a n1-standard-2 instance.

## Run Tests GPU

The only tests maintained are related to intra-fid calculation and some new losses:

```
CUDA_VISIBLE_DEVICES=0 python tensorflow_gan/python/eval/eval_utils_test.py
CUDA_VISIBLE_DEVICES=0 python tensorflow_gan/python/eval/classifier_metrics_test.py
CUDA_VISIBLE_DEVICES=0 python tensorflow_gan/python/losses/other_losses_impl_test.py
```

## Contributions

PRs are welcome! See `/gan/DEVREADME.md` for more info.
