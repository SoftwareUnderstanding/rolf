# MLP-Mixer
My project is using the TensorFlow framework to implement the [MLP-Mixer model](https://arxiv.org/abs/2105.01601). <br>
Give me a star :star2: if you like this repo.

### Model Architecture
<p align = "center"> 
<img src = "image/mlp_mixer.png">
</p>

### Author
<ul>
    <li>Github: <a href = "https://github.com/Nguyendat-bit">Nguyendat-bit</a> </li>
    <li>Email: <a href = "nduc0231@gmai.com">nduc0231@gmail</a></li>
    <li>Facebook: <a href = "https://www.facebook.com/dat.ng48/">Nguyễn Đạt</a></li>
    <li>Linkedin: <a href = "https://www.linkedin.com/in/nguyendat4801">Đạt Nguyễn Tiến</a></li>
</ul>

## I.  Set up environment
- Step 1: Make sure you have installed Miniconda. If not yet, see the setup document <a href="https://docs.conda.io/en/latest/miniconda.html">here</a>


- Step 2: `cd` into `MLP-Mixer` and use command line
```
conda env create -f environment.yml
```

- Step 3: Run conda environment using the command

```
conda activate mlp-mixer
``` 

## II.  Set up your dataset

<!-- - Guide user how to download your data and set the data pipeline  -->
1. Download the data:
- Download dataset [here](http://download.tensorflow.org/example_images/flower_photos.tgz)
2. Extract file and put folder ```train``` and ```validation``` to ```./data``` by using [splitfolders](https://pypi.org/project/split-folders/)
- train folder was used for the training process
- validation folder was used for validating training result after each epoch

This library use ImageDataGenerator API from Tensorflow 2.0 to load images. Make sure you have some understanding of how it works via [its document](https://keras.io/api/preprocessing/image/)
Structure of these folders in ```./data```

```
train/
...daisy/
......daisy0.jpg
......daisy1.jpg
...dandelion/
......dandelion0.jpg
......dandelion1.jpg
...roses/
......roses0.jpg
......roses1.jpg
...sunflowers/
......sunflowers0.jpg
......sunflowers1.jpg
...tulips/
......tulips0.jpg
......tulips1.jpg
```

```
validation/
...daisy/
......daisy2000.jpg
......daisy2001.jpg
...dandelion/
......dandelion2000.jpg
......dandelion2001.jpg
...roses/
......roses2000.jpg
......roses2001.jpg
...sunflowers/
......sunflowers2000.jpg
......sunflowers2001.jpg
...tulips/
......tulips2000.jpg
......tulips2001.jpg
```

## III. Train your model by running this command line

Review training on colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J_xSQ-9hhOv_J-wGyYAxrusDFw7-phb0?usp=sharing)


Training script:


```python

python train.py --train-folder ${link_to_train_folder} --valid-folder ${link_to_valid_folder} --epochs ${epochs}

```


Example:

```python

python train.py  --train-folder ./data/train --valid-folder ./data/val --epochs 100 

``` 

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- `model-save`: Where the model after training saved
- `batch-size`: The batch size of the dataset
- `lr`: The learning rate
- `image-size`: The image size of the dataset
- `Dc`: Hidden units of channel-mixing. It was mentioned in the paper on [page 3](https://arxiv.org/pdf/2105.01601.pdf)
- `Ds`: Hidden units of token-mixing. It was mentioned in the paper on [page 3](https://arxiv.org/pdf/2105.01601.pdf)
- `mixer-layer`: The number of mixer-layer 
## IV. Predict Process
If you want to test your single image, please run this code:
```bash
python predict.py --test-file ${link_to_test_image}
```




## V. Feedback
If you meet any issues when using this library, please let us know via the issues submission tab.



