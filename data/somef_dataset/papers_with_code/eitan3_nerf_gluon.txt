# nerf-gluon
#### A Mxnet re-implementation
### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934)


[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution

<p align="center">
    <img src="assets/pipeline.jpg"/>
</p>

A Mxnet re-implementation of [Neural Radiance Fields](http://tancik.com/nerf). Used some code from the repositories [bmild/nerf](https://github.com/bmild/nerf), [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), [krrish94/nerf-pytorch](https://github.com/krrish94/nerf-pytorch), Thank you for the original code!

## What is a NeRF?

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

## Installation
install mxnet 2.0.0 from here https://dist.mxnet.io/python/all <br>
Install all packages with pip
```
pip install -r requirements.txt
```

## Training
* Download the dataset from [this](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (contain real world data and synthetic data)<br>
* Extract the data to a chosen folder. For this example I extract it to C:\Datasets\Nerf\ and use 'lego' dataset for training (folder path: C:\Datasets\Nerf\nerf_example_data\nerf_synthetic\lego\)<br>
* Create a folder inside the chosen folder and name it cache (C:\Datasets\Nerf\nerf_example_data\nerf_synthetic\lego\cache) <br>
* Run cache_dataset.py to preprocess the dataset
```
python cache_dataset.py --datapath 'C:\Datasets\Nerf\nerf_example_data\nerf_synthetic\lego\' --savedir 'C:\Datasets\Nerf\nerf_example_data\nerf_synthetic\lego\cache' --type blender --sample_all
```
* Edit 'lego.yml' config file inside 'configs' folder and change 'cachedir' to the folder contain the cache files (for this example 'C:\Datasets\Nerf\nerf_example_data\nerf_synthetic\lego\cache')
* Run train.py to start training
```
python train_nerf.py --config './configs/lego.yml'
```

##  Citation
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```