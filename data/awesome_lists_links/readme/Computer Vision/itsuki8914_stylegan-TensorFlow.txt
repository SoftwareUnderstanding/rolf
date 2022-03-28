# StyleGAN-TensorFlow
A implementation of StyleGAN using Tensorflow

This implementation is on experiment and has many issues.

official paper: https://arxiv.org/abs/1812.04948

official implementation: https://github.com/NVlabs/stylegan

This code is based on [my PGGAN implementation.](https://github.com/itsuki8914/PGGAN-TensorFlow)

<img src = 'examples/top.png' width=1280>

## Usage

1. Download ffhq-dataset from [here.](https://github.com/NVlabs/ffhq-dataset)

2. Put images1024x1024 in ffhq_dataset and thumbnails128x128 in ffhq_dataset128.

like this

```
...
│
├── ffhq_dataset
│     ├── 00000.png
│     ├── 00001.png
│     ├── ...
│     └── 69999.png
├── ffhq_dataset128
│     ├── 00000.png
│     ├── 00001.png
│     ├── ...
│     └── 69999.png 
├── main.py
├── model.py
...
```

3. Train StyleGAN.

```
python main.py
```

How long does it take to train using RTX 2070,
```
64x64     1d00h
128x128   2d00h
256x256   3d18h
512x512   6d13h(estimated)
1024x1024 unknown
```

4. After training, inference can be performed.

to draw uncurated images,
```
python pred.py -m uc
```

<img src = 'examples/uc_ffhq.png' width=1280>

to draw truncation trick images,
```
python pred.py -m tt
```

<img src = 'examples/tt_ffhq.png' width=1280>

to draw style mixing images,
```
python pred.py -m sm
```

<img src = 'examples/sm_ffhq.png' width=1280>

## Other Results

### Anime faces

uncurated
<img src = 'examples/uc_anime.png' width=1280>

truncation trick
<img src = 'examples/tt_anime.png' width=1280>

style mixing
<img src = 'examples/sm_anime.png' width=1280>

## Issues

As mentioned at the beginning, this implementation has problems.

### First layer noise

<img src = 'examples/1st_noise.png' width=1280>

Diversity and quality are too sensitive to 1st noise.

The above figure is from the left

Use all noise

without 1st layer noise

without 1st and 2nd layers

without all noise 

Because of this, style mixing and truncation trick cannot use 1st noise.

Disabling 1st noise results slight improvement in quality and loss of diversity.

below images are style mixing and trucation trick using noise of 1st layer.

<img src = 'examples/sm_1stnoise.png' width=1280>

<img src = 'examples/tt_1stnoise.png' width=1280>

### Initial interpolation for each stage

<img src = 'examples/init_stage.png' width=480>

At the beginning of the stage, the kernels are unlearned, so interpolation is not performed well, this leads to quality degradation?

There are some other problems and I will continue experiments.
