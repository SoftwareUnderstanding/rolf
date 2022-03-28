# Unpaired Image to Image translation using CycleGAN

<p align='center'>
  <img src='g3doc/images/horse_zebra/horse.gif'>
  <img src='g3doc/images/horse_zebra/horse2zebra.gif'>
  <img src='g3doc/images/horse_zebra/zebra.gif', width=372>
  <img src='g3doc/images/horse_zebra/zebra2horse.gif', width=372>
  <br>
  Translating an animated gif file of a running horse(zebra) into a running zebra(horse) using CycleGAN.
</p>


<p align='center'>
  <img src='g3doc/images/cyclegan_arch.png'>
  <br>
  CycleGAN architecture: Figure 3 of Zhu et al. 2017(https://arxiv.org/abs/1703.10593)
</p>




This repo documents my tensorflow implementation of CycleGAN for performing unpaired image-to-image translation (e.g. object transfiguration, style transfer). As shown in the diagram, it involves two pairs of Discriminator ($D_X$ and $D_Y$) and Generator ($G$ and $F$), where $D_X$ tries to distinguish real images in domain X from fake images in domain X, which are generated from source images in domain Y using $F$ (i.e. $F: Y->X$); likewise, $D_Y$ tries to distinguish real images in domain Y from fake images in domain Y, which are generated from source images in domain X using $G$ (i.e. $G: X->Y$). The goal is that 
* The probability distribution of real and fake images in domain X (or Y) are indistinguishable from each other.
* Translating from domain X (Y) to domain Y (X), then back to domain X (Y) should be close to the identity mapping.

# Usage
## Clone the Repo

```
git clone git@github.com:chao-ji/image-translation.git
```

## Data 
You can download the images from the author's [website](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets) (e.g. `horse2zebra`, `photo2monet` etc.). Or you can prepare your own datasets (i.e. images in domain X and domain Y). 

Before training the model, the images must be converted into `TFRecord` format. Suppose the images in domain X and domain Y are located in directories `domain_x_images` amd `domain_y_images`,

run

```
python create_dataset.py \
  --X_images_dir=domain_x_images \
  --Y_images_dir=domain_y_images
```
The images in `domain_x_images` amd `domain_y_images` will be stored in `x.tfrecord` and `y.tfrecord`.

## Train
To train the model, run
```
python run_trainer.py \
  --domain_x_filenames=x.tfrecord \
  --domain_y_filenames=y.tfrecord \
  --num_minibatches_epoch=N \
```
where `N` is number of image pairs to be fed to the model in one epoch (typically you can set it to the size of domain X or domain Y).

The learning rate stays constant in first half of training epochs, and linearly decays to zero in the second half. Run `python run_trainer.py --help` for more info.

## Inference
Once the model is trained, you can perform image translation given images from domain X or domain Y.

Suppose the images in domain X and domain Y are located in directories `domain_x_images` amd `domain_y_images`, and `model-200` is the checkpoint file holding trained variables. Run

```
python run_inferencer.py \
  --ckpt_filename=model-200 \
  --domain_x_dir=domain_x_images \
  --domain_y_dir=domain_y_images \
  --output_dir=/path/to/output_dir
```

The translated images will be located in `/path/to/output_dir`.

# Results
## Object Transfiguration
### Horse to (and from) zebra 
I trained the model on `horse2zebra` dataset (1067 horse images and 1334 zebra images) for 200 epochs. Below are 5 horse2zebra translations and 5 zebra2horse translations (Left: real, Middle: fake, Right: cycle) 
<p align='center'>
  <img src='g3doc/images/horse_zebra/n02381460_1160.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02381460_4240.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02381460_4640.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02381460_5090.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02381460_8900.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02391049_1430.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02391049_3010.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02391049_400.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02391049_7860.jpg.png'>
  <img src='g3doc/images/horse_zebra/n02391049_9680.jpg.png'>
</p>

The translated images as training progresses:

<p align='center'>
  <img src='g3doc/images/horse_zebra/horse2zebra_progress.gif'>
  <img src='g3doc/images/horse_zebra/zebra2horse_progress.gif'>
</p>

# Reference
* CycleGAN, [arxiv1703.10593](https://arxiv.org/abs/1703.10593)
* [Official implementation of CycleGAN](https://junyanz.github.io/CycleGAN/)
