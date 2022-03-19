# CIL_project
Project for the ETH Computational Intelligence Lab.

Authors: Sven Kellenberger, Hannes Pfammatter, Till Schnabel, Michelle Woon

Group: Galaxy Crusaders

## Table of Contents

1. [Requirements](#requirements)
2. [Trained Models](#trained_models)
3. [Image Generation Task](#image_generation)

   [stars_extractor](#stars_extractor)
   
   [Adhoc_generator](#adhoc)
   
   [DCGAN](#dcgan)
   
   [cDCGAN](#cdcgan)
   
   [VAE_stars](#vae)
   
   [AE_plus_KMeans](#ae_plus_kmeans)
   
   [Image Scorer](#image_scorer)
   
4. [Similarity Scorer Task](#similarity_task)

   [Classifier](#classifier)
   
   [RandomForest](#random_forest)
   
5. [Report](#report)

6. [generated_images]{#generated_images}


<a name="requirements"/>

## Requirements

Install all requirements with

    pip install -r requirements.txt

After installing all requirements, head into the `utils/` folder and run `python setup.py build_ext -i`
to compile the cython files there.


<a name="trained_models"/>

## Trained Models
For most models used in this project, there is a `reference_run` folder inside the corresponding directory, which contains the trained model and some additional basic information like the configuration.  For the random forest, there is an additional `reference_run_baseline` folder that contains the baseline model mentioned in the report.


<a name="image_generation"/>

## Image Generation Task

<a name="stars_extractor"/>

### stars_extractor

This project only contains some basic scripts. To extract stars from the original images, run

    python stars_extractor.py --img_dir=/dir/with/large/images --target_dir=/dir/to/save/star/patches 

To filter the original star images, run

    python create_dir_for_labeled_star_images.py --dataset_dir=/path/to/cosmology_aux_data_170429 --target_dir=/dir/to/save/filtered/images

where you may also specify whether to filter the labeled images to only keep those with label 1 by setting `--kind=labeled`, or to filter the scored images to only keep those with score above a custom number by setting `--scored_thresh=your_scored_threshold`.

To measure and approximate an unsigned-integer-bound gaussian distribution of all kinds of stars in the images, run

    python stars_clustered_distribution.py --unclustered_stars_dir=/dir/containing/unclustered/stars --clustered_stars_dir=/dir/containing/clustered/stars

<a name="adhoc"/>

### Adhoc_generator

This adhoc method randomly places stars that it has detected from the given labelled images and
places them randomly onto a black image.

    python Adhoc.py --data_path=/path/to/data

<a name="dcgan"/>

### DCGAN

A large DCGAN for the generation of galaxy images. The model can be trained with
    
    python dcgan_train.py --data_directory=/path/to/dataset

After 80 epochs, the weights of the generator are saved every 5 epochs inside `/ckpt`.

To generate galaxy images with the pretrained reference model run:

    python generate_galaxy_images.py --data_directory=/path/to/dataset

The generated 1000x1000 galaxy images are subsequently saved inside `/generated`. 

<a name="cdcgan"/>

### cDCGAN

    python gan.py --dataset-dir=/path/to/dataset

To train a conditional model on the 28x28 star patches, adjust the config.yaml s.t. the variable `conditional`is set to `True` and `model_kind`to `4`. Also make sure that the provided data set contains a folder for each category with the corresponding images inside. For unconditional training on
the 28x28 star patches, set `conditional` to `False` and `model_kind` to `3`, and make sure that the provided path
to the data set directly contains the images. The results will be saved in a new folder inside the `checkpoints` directory.

To generate complete images using the trained cDCGAN and also score them, run 

    python create_complete_images.py

The algorithm will loop infinitely by default to find a good distribution. Set `--find_good_latents` to `False` if you wish to simply create and score some images without the infinite loop using the saved distribution. Provide the path to a cDCGAN checkpoint as argument `--checkpoint_path` if you wish to use another than the default one. However, the files `gen_config.json` and `config.yaml` must be in the same directory as the custom checkpoint.

<a name="vae"/>

### VAE_stars

A variational autoencoder model for star image generation. The model can be trained with
    
    python star_vae_train.py --data_directory=/path/to/dataset

The weights of the generative model (decoder) are subsequently saved inside `/ckpt_generative`.

To generate star images with the pretrained reference model run:

    python generate_star_images.py --data_directory=/path/to/dataset

The generated 28x28 star images are subsequently saved inside `/generated`.

To create complete galaxy images, run then:

    python generate_complete_images.py --data_directory=/path/to/dataset

The generated 1000x1000 galaxy images are subsequently saved inside `/generated`.

<a name="ae_plus_kmeans"/>

### AE_plus_KMeans

The purpose of this project is to find a compact representation for a dataset in order to cluster the stars using lower-dimensional data.
First, an autoencoder is trained to find said compact representation. Afterwards, k-means is applied to the encoder's latent
code of the images to cluster them.

For the autoencoder, run

    python ae.py --image_dir=dir/with/star/images

where `dir/with/star/images` is the directory containing the star patches of size 28x28 directly. A trained model of the
encoder is saved in a separate directory inside the `checkpoints` folder. Provide the path to this encoder model to the
k-means script as argument. Run

    python kmeans.py --econder_path=/path/to/encoder_config.json

The clustered images are saved to a separate directory inside `images/clustered_stars` if not specified otherwise via
the `--target_dir` argument.

<a name="image_scorer"/>

### Image Scorer

Use the file `cDCGAN/img_scorer.py` to score an arbitrary image of size 1000x1000 or a folder containing images of size
1000x1000. Provide the path to either the image or the folder via the argument `--path`. You will get as output a score
approximated by both the CNN and RF and additionally their mean. Run

    python img_scorer.py --path=/path/to/images

<a name="similarity_task"/>

## Similarity Scorer Task

<a name="classifier"/>

### Classifier

A similarity scorer using a Convolutional Neural Network.

Training:

    python classifier.py --dataset_dir=/path/to/cosmology_aux_data_170429

Prediction:

    python classifier.py --test_on_query=True --dataset_dir=/path/to/cosmology_aux_data_170429 --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001

Where `/path/to/checkpoint/cp####.ckpt.data-00000-of-00001`
is a valid path to the checkpoint and `####` is replaced with the checkpoint number.

<a name="random_forest"/>

### RandomForest

This model takes two command line arguments which are required:

- `--data-directory`: The directory where the dataset is stored.
- `--dump-direcotry`: The directory where all generated data should be stored. This directory
will be created if it doesn't exist yet.

All other options can be set in the config file.

To train the model, head into the `RandomForest/` subdirectory and run:

    python random_forest.py --data-directory=/path/to/the/data/ --dump-directory=/path/to/dump/directory/
    
<a name="report"/>
    
## Report
This directory contains the LaTeX files as well as the compiled pdf of the report.


<a name="generated_images"/>

## generated_images
This directory contains our final generated cosmology images.
