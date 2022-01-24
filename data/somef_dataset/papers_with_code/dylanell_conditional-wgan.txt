# conditional-gan

| ![](conditional_gan/artifacts/gen.gif) |
| :-: |
| *Generator outputs with 20 constant label inputs during training. Column labels can be found in [`gen_gif_cols.txt`](https://github.com/dylanell/conditional-gan/blob/main/conditional_gan/artifacts/gif_cols.txt)* |

This project contains a PyTorch implementation of a Conditional [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) trained on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/), combined with a simple model serving API for the GAN generator network.

Compared to a regular GAN, the generator of a conditional GAN takes two inputs; the original randomly sampled input vector plus an additional randomly sampled one-hot vector. This additional one-hot vector takes on the role of representing the "class" of a generated sample while the z vector is left to represent the "style". One can therefore individually control both the "style" and the "label" for generated samples, respectively, by changing these two generator input vectors.

Instead of training a "vanilla" conditional GAN in this project, we follow some of the techniques used in [this paper](https://arxiv.org/pdf/1706.07068.pdf) to observe the effects of adding an objective that promotes "creativity" in the generator model for a GAN, dubbed the "Creative GAN (CAN)". Unlike a standard conditional GAN where the generator model is trained to minimize classifier error for conditionally generated images labeled by the "class" vector inputs, the CAN is trained to maximize classifier error for all generated samples. The motivation is that by maximally "confusing" the classifier, while still fooling the critic into thinking a generated sample is real, the generator will learn to "create" new instances that "fall within the gaps" of the classifier, which is simultaneously trained on real images. The CAN paper is trained and evaluated on a dataset of art images, which can be a little subjective when it comes to defining what is "creatively" novel. For this project, we would like to explore the effects of adding this "creativity" objective for a dataset in which class differences are very succinct, like MNIST.

Training the CAN on MNIST results in a generator model that doesn't generate very "creative" looking digits at all. This result is somewhat unsurprising if you think conceptually about what it means to maximize the confusion for a classifier trained on a dataset of real samples. In the CAN paper, the authors aim to achieve this by minimizing the cross entropy between the classifier output on generated images and the uniform class distribution ("all-hot" label vector), resulting in an optimization problem that essentially asks the generator to create realistic looking digits (from the critic's perspective) that look like all digits at once. This is a pretty difficult task for the generator to solve.

Instead of following the "maximal confusion" objective from the CAN paper directly, we relax the rules slightly to allow for randomly sampled "multi-hot" or "k-hot" labels for generated samples, where k can be anything from 1 to the number of classes. Additionally, we utilize the conditional GAN architecture so that we can control these "multi-hot" conditional inputs, therefore controlling the "creativity" at the output of the generator model. To do this, we parameterize a custom "k-hot" categorical meta-distribution by a "pan-classness" parameter `pan`, which controls the tendency to either sample more "one-hot" distributions (standard conditional GAN training) or more "all-hot" distributions (standard CAN training). We can therefore train a CAN with "k-hot" generator labels where k is more often a moderate value less than the total number of classes, but also not always just 1. The gif above shows a conditional GAN trained in this fashion, where some of the conditional label vectors are "2-hot" label.

### Project Structure:

Python files that define the architecture and training scripts for the conditional GAN model are located within the `conditional_gan` project directory. The `server` directory contains Python files for building and running the generator model serving API. The full project structure tree is below.  

```
conditional-gan/
├── conditional_gan
│   ├── artifacts
│   │   ├── classifier.pt
│   │   ├── critic.pt
│   │   ├── generator.pt
│   │   ├── gen.gif
│   │   └── gif_cols.txt
│   ├── config.yaml
│   ├── dashboard.py
│   ├── data
│   │   ├── datasets.py
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── __init__.py
│   ├── modules.py
│   ├── train.py
│   └── util
│       ├── distributions.py
│       └── __init__.py
├── docker-compose.yaml
├── Dockerfile
├── README.md
├── requirements.txt
└── server
    ├── api.py
    ├── __init__.py
    └── wrappers.py
```

### Local Setup:

Runtime:

```
Python 3.8.5
```

Install Python requirements:

```
$ pip install -r requirements.txt
```

### Image Dataset Format:

This project assumes you have the MNIST dataset downloaded and preprocessed locally on your machine in the format described below. My [dataset-helpers](https://github.com/dylanell/dataset-helpers) Github project also contains tools that perform this local configuration automatically within the `mnist` directory of the project.

The MNIST dataset consists of images of written numbers (0-9) with corresponding labels. The dataset can be accessed a number of ways using Python packages (`mnist`, `torchvision`, `tensorflow_datasets`, etc.), or it can be downloaded directly from the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). In order to develop image-based data pipelines in a standard way, we organize the MNIST dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. This "generic image dataset format" is summarized by the directory tree structure below.

```
mnist_png/
├── test
│   ├── test_image_01.png
│   ├── test_image_02.png
│   └── ...
├── test_labels.csv
├── train
│   ├── train_image_01.png
│   ├── train_image_02.png
│   └── ...
└── train_labels.csv
```

Each labels `csv` file has the format:

```
Filename, Label
train_image_01.png, 4
train_image_02.png, 7
...
```

If you would like to re-use the code here to work with other image datasets, just format any new image dataset to follow the outline above and be sure to update corresponding parameters in `config.yaml`.

### Training:

To train the model, navigate to the `conditional_gan` directory and run:

```
$ python train.py
```

The training script will generate model artifacts to the `artifacts/` directory. Configuration and training parameters can be controlled by editing `config.yaml`.

### Generator Model Dashboard:

The `dashboard.py` script uses the [Dash](https://dash.plotly.com/) package to create a simple generator GUI, allowing you to manually control generator inputs and visualize their corresponding generated images. Navigate to the `conditional_gan` directory and run the following command to access the generator dashboard at `http://localhost:8050/` in a browser.

```
$ python dashboard.py
```

Move the sliders to generate "one-hot" (or "multi-hot") label vectors, or press the "Sample New Style" button to sample new style vectors. The dashboard automatically generates the corresponding new output image any time these inputs are changed. We can try to control the "creativity" at the output of the generator by providing it with "multi-hot" labels to see if it can create "new" digits from combined generator features.   

### Serving:

This project also uses [FastAPI](https://fastapi.tiangolo.com/) to setup a model serving API for a pre-trained generator model. Swagger UI interactive API documentation can be viewed at `http://localhost:8080/docs` on a browser.

Serve with Python:

Navigate to the `server` directory and run the following command to spin up the API server on `http://localhost:8080/`.

```
$ python api.py
```

Serve with Docker:

Make sure you have [Docker](https://www.docker.com/) installed along with the [docker-compose](https://docs.docker.com/compose/install/) cli. Run the following command from the top level of this project directory to spin up a container that runs the API server on `http://localhost:8080/`. The first run may take a bit to build the new Docker image with all the Python package dependencies.

```
$ docker-compose up
```

### References:

1. [Improved Wasserstein GAN](https://arxiv.org/pdf/1704.00028.pdf)
2. [Creative Adversarial Network (CAN)](https://arxiv.org/pdf/1706.07068.pdf)
