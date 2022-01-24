# Texture Networks + Instance normalization: Feed-forward Synthesis of Textures and Stylized Images

In the paper [Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417) we describe a faster way to generate textures and stylize images. It requires learning a feedforward generator with a loss function proposed by [Gatys et al.](http://arxiv.org/abs/1505.07376). When the model is trained, a texture sample or stylized image of any size can be generated instantly.

[Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis](https://arxiv.org/abs/1701.02096) presents a better architectural design for the generator network. By switching `batch_norm` to `Instance Norm` we facilitate the learning process resulting in much better quality.



This also implements the stylization part from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

You can find an oline demo [here](https://riseml.com/DmitryUlyanov/texture_nets) (thanks to RiseML). 
# Prerequisites
- [Torch7](http://torch.ch/docs/getting-started.html) + [loadcaffe](https://github.com/szagoruyko/loadcaffe)
- cudnn + torch.cudnn (optionally)
- [display](https://github.com/szym/display) (optionally)

Download VGG-19.
```
cd data/pretrained && bash download_models.sh && cd ../..
```

# Stylization
<!-- 
Content image|  Dalaunay | Modern 
:-------------------------:|:-------------------------:|:------------------------------:
![](data/readme_pics/karya.jpg " ") | ![](data/readme_pics/karya512.jpg  " ")| ![](data/readme_pics/karya_s_mo.jpg  " ")
 -->
![](data/readme_pics/all.jpg " ")

### Training

#### Preparing image dataset

You can use an image dataset of any kind. For my experiments I tried `Imagenet` and `MS COCO` datasets. The structure of the folders should be the following:
```
dataset/train
dataset/train/dummy
dataset/val/
dataset/val/dummy
```

The dummy folders should contain images. The dataloader is based on one used in[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). 

Here is a quick example for MSCOCO: 
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip train2014.zip
unzip val2014.zip
mkdir -p dataset/train
mkdir -p dataset/val
ln -s `pwd`/val2014 dataset/val/dummy
ln -s `pwd`/train2014 dataset/train/dummy
```

#### Training a network

Basic usage:
```
th train.lua -data <path to any image dataset>  -style_image path/to/img.jpg
```

These parameters work for me: 
```
th train.lua -data <path to any image dataset> -style_image path/to/img.jpg -style_size 600 -image_size 512 -model johnson -batch_size 4 -learning_rate 1e-2 -style_weight 10 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2
```
Check out issues tab, you will find some useful advices there. 

To achieve the results from the paper you need to play with `-image_size`, `-style_size`, `-style_layers`, `-content_layers`, `-style_weight`, `-tv_weight`. 

Do not hesitate to set `-batch_size` to one, but remember the larger `-batch_size` the larger `-learning_rate` you can use.   

### Testing

```
th test.lua -input_image path/to/image.jpg -model_t7 data/checkpoints/model.t7
```

Play with `-image_size` here. Raise `-cpu` flag to use CPU for processing. 

You can find a **pretrained model** [here](https://yadi.sk/d/GwL9jNJovBwQg). It is *not* the model from the paper.

## Generating textures

soon
<!-- ## Train texture generator

### Train

This command should train a generator close to what is presented in the paper. It is tricky, the variance in the results is rather high, many things lead to degrading (even optimizing for too long time).
```
th texture_train.lua -texture data/textures/red-peppers256.o.jpg -model_name pyramid -backend cudnn -num_iterations 1500 -vgg_no_pad true -normalize_gradients true -batch_size 15
```
The generator will fit the texture

![Texture](data/textures/red-peppers256.o.jpg)

And here is a sample of size `512x512` after learning for 700 iterations:

![Sample](data/readme_pics/peppers_sample.png)


You may also explore other models. We found `pyramid2` requires bigger `learning rate` of about `5e-1`. To prevent degrading noise dimensionality should be increased: `noise_depth 16`. It also converges slower.

This works good for me:
```
th texture_train.lua -texture data/textures/red-peppers256.o.jpg -gpu 0 -model_name pyramid2 -backend cudnn -num_iterations 1500 -vgg_no_pad true -normalize_gradients true -learning_rate 5e-1 -noise_depth 16
```

- `vgg_no_pad` corresponds to padding option used in VGG. If set, padding mode = `valid`.

The samples and loss plot will appear at `display` web interface.

### Sample

A sample from above can be obtained with
```
th texture_sample.lua -model data/out/model.t7 -noise_depth 3 -sample_size 512
```
`noise_depth` should correspond to `noise_depth` used when training.

## Stylization


### Prepare

We used ILSVRC2012 validation set to train a generator. One pass through the data was more than enough for the model described in the paper.

Extract content from `relu4_2` layer.
```
th scripts/extract4_2.lua -images_path <path/ILSVRC2012>
```
### Train

Use this command to learn a generator to stylize like in the next example.
```
th stylization_train.lua -style_image data/textures/cezanne.jpg -train_hdf5 <path/to/generated/hdf5> -noise_depth 3 -model_name pyramid -normalize_gradients true -train_images_path <path/to/ILSVRC2012> -content_weight 0.8

```
### Process

Stylize an image.
```
th stylization_process.lua -model data/out/model.t7 -input_image data/readme_pics/kitty.jpg -noise_depth 3
```
Again, `noise_depth` should be consistent with training setting.

### Example

![Cezanne](data/textures/cezanne.jpg)

![Original](data/readme_pics/kitty.jpg)

![Processed](data/readme_pics/kitty_cezanne.jpg)

#### Variations
We were not able to archive similar results to original parer of L. Gatys on artistic style, which is partially explained by balance problem (read the paper for the details). Yet, while not transferring the style exactly as expected, models produce nice pictures. We tried several hacks to redefine the objective function, which could be more suitable for convolutional parametric generator, none of them worked considerably better, but the results were nice.

For the next pair we used a generator, trained using 16 images only. It is funny, that it did not overfit. Also, in this setting the net does not degrade for much longer time if zero padding is used. Note that, tiger image was not in the train set.

![Tiger](data/readme_pics/tiger.jpg)

![Tiger_processed](data/readme_pics/tiger_starry.jpg)
Using "Starry night" by Van Gogh. It takes about quarter of second to process an image at `1024 x 768` resolution.


In one of the experiments the generator failed to learn Van Gogh, but went very stylish.

![Pseudo](data/readme_pics/pseudo.png)

This model tried to fit both texture and content losses on a fixed set of 16 images and only content loss on the big number of images.
 -->

# Hardware
- The code was tested with 12GB NVIDIA Titan X GPU and Ubuntu 14.04.
- You may decrease `batch_size`, `image_size` if the model do not fit your GPU memory.
- The pretrained models do not need much memory to sample.

# Credits

The code is based on [Justin Johnson's great code](https://github.com/jcjohnson/neural-style) for artistic style.

The work was supported by [Yandex](https://www.yandex.ru/) and [Skoltech](http://sites.skoltech.ru/compvision/).
