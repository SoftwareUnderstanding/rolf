# Detection and classification of ancient Japanese Kuzushiji characters

# Table of Contents

- [Aim](#aim)
- [Requirements](#requirments)
- [Repository Structure](#repository-structure)
- [Project Architecture](#project-architecture)
- [Main Components](#main-components)
    - [download_data.py](#download_data.py)
    - [dataloader.py](#dataloader.py)
    - [model.py](#model.py)
    - [main.py](#main.py)
- [Suggested Usage](#suggested-usage)
- [Models](#models)
    - [Detection](#detection)
    - [Classification](#classification) 
        - [Baseline ConvNet](#baseline-convnet)
        - [ResNet18](#resnet18)
        - [MobileNetV3](#mobilenetv3)
- [Results](#results)
- [References](#references)

# Aim

The main goal of this project has been to develop a model (models) that would perform detection and classification of ancient Japanese characters (Kuzushiji cursive script), in which classification consists of mapping the Kuzushiji characters to their modern Japanese counterparts. 

The main motivation behind the project choice has been to utilize artificial intelligence tools to contribute to a wider ongoing research aimed at making ancient Japanese culture and history more available to people[1]. Sources written in Kuzushiji cannot be read nor appreciated without appropriate translation by anyone except only a small number of experts. Being able to make the ancient Japanese heritage more accessible to a wider public seemed like a fantastic real-life application of Machine Learning.

Data for the project has been taken from the Kaggle competition[2] aimed at improving the current models developed for Kuzushiji recognition. 

# Requirements

- Python 3.8
- tqdm
- pandas
- requests
- Tensorflow 2.4
- tensorflow-addons


# Repository Structure

    .
    ├── config/                 # Config files
    ├── data/                   # Dataset path
    ├── notebooks/              # Prototyping
    ├── scripts/                # Download dataset and miscelaneus scripts
    ├── trained_models/         # Trained weights
    ├── dataloader.py           # Data reader, preprocessing, batch iterator
    ├── download_dataset.py     # Data downloader
    ├── main.py                 # Running an experiment (different modes below)
    └── model.py                # Defines model
    

# Project Architecture

![Simplified architecture of the project](./figures/arch.png?raw=true)

# Main Components

## download_data.py

Script for downloading the data available on Kaggle website[2]. Zip files are downloaded directly to the data/ directory.

## dataloader.py

Script for unpacking the zipped data and creating TensorFlow.records input pipelines for the detection and classification tasks, as well as the config json file to be used later by `main.py`. Users need to specify the kind of model for which they intend to use the data using the flags (names are self-explanatory):

- `--detector`
- `--classifier`

`dataloader.py` has a set of default parameters to be saved in the config file, but accepts custom values through the appropriate flags. See --help for more information.

## model.py

Script containing the detection and classification models used in `main.py`. At the moment, detection is performed using the CenterNet[3] model only. For classification, users can use the `--classifierName` flag to choose one of the currently supported models: ConvNetBaseline (custom default), ResNet(18 or 34)[4] or MobileNetV3 Large[5].

## main.py

Script for running the detection and classification experiments. 

The following modes are set through the `--mode` flag:

- `train` : Fit the estimator using the training data.
- `evaluate` : Evaluate on the evaluation data (requires previous training of both models or a path to saved models specified with the flags `--detectorPath` and `--classifierPath`).

# Suggested Usage

The suggested usage of the project's resources available here is as follows (the users are however free to use them at their will):

1. Install requirements.

    ```shell
    pip install -r requirements.txt
    ```

2. Download the raw data set[2]:

    ```shell
    python download_dataset.py
    ```
    
3. Unpack the zipped data and pre-process it to create a TensorFlow input pipeline and a config json file used by `main.py` for a desired task using `dataloader.py` (remember to specify the kind of model for which you intend to use the data using appropriate flag):

    ```shell
    python dataloader.py --detector
    ```

4. Finally, train the desired model using `main.py`:

    ```shell
    python main.py --detector --mode train --numEpochs 20 --gpu 1 --minLr 1e-4
    ```
    
The model hyperparameters should be supplied through appropriate flags. Use --help for more information.

# Models

## Detection

Below we present sample images showing the results of our experiments regarding the detection task.

- Learning curves obtained from the training process of the CenterNet detection model:
![Learning curves from the CenterNet detection model](./figures/centernet_curves.png?raw=true)

The model was trained for over 150 epochs with a batch size of 1024 using the Adam optimizer. We applied a reduce on plateau learning rate schedule starting from 0.01, and cyclical learning restarting about every 60 epochs.

- Examples of the input and first three output channels from the trained CenterNet:

![Positions of characters on sample page predicted by CenterNet](./figures/positions.png?raw=true)

The first channel is a heat map of the predicted center of the character. The brightness of the dots in the second and third channel is proportional to the predicted size of the object horizontally and vertically, respectively.
- Predicted bounding boxes obtained from the trained CenterNet:

![Bounding Boxes generated with CenterNet](./figures/boxes.png?raw=true)

Overall, the model performs nicely for small and close to average characters (left. Bear in mind that the small anotations on the sides of the columns are meant to be ignored by the model), but as can be seen (right), it can fail for unusually large characters, as these were rather uncommon on the train set.

## Classification

Below we present sample images showing the results of our experiments regarding the classification task.

### Baseline ConvNet

The baseline convolutional neural network we have developed for the classification task has the following architecture:

![Baseline Convolutional Model Architecture](./figures/baseline_arch.png?raw=true)


Training of the baseline convolutional net has been performed with a constant learning rate of 0.001, categorical cross-entropy loss, Adam optimizer, batch size of 512 and 20 epochs.

- Sample learning curves obtained from the Baseline Convolutional classification model:

![Learning curves from the ConvNetBaseline classification model](./figures/convnet_curves.png?raw=true)


### ResNet18

Aside from our own simple baseline model, we have tried using the well known ResNet model, more specifically the ResNet18 architecture. The model has been implemented by hand. The training process was performed with a reduce on plateau learning rate schedule, categorical cross-entropy loss, Adam optimizer, batch size of 256 and 100 epochs.

- Sample learning curves obtained from the ResNet18 classification model:

![Learning curves from the uniweighted ResNet18 classification model](./figures/resnet_unweighted.png?raw=true)


### MobileNetV3

The core of the MobileNetV3 Large[5] model available in the keras.applications package with an additional densely connected layer (units=1024) followed by batch normalization, leaky ReLU (alpha=0.1) and dropout (rate=0.25) layers before the final output layer with a suitable number of outputs (4206) has been used for the purposes of our experiments. 

The training process has been performed with a random initialization of model weights, reduce on plateau learning schedule, minimal learning rate of 1e-4, categorical cross-entropy loss, Adam optimizer, batch size of 4096 and 100 epochs. For this model we additionally used the class weighting scheme described in [6] to try to counter the considerable class imbalance present in the data set.

- Sample learning curves obtained from the MobilenetV3 classification model:

![Learning curves from the MobileNetV3 Large classification model](./figures/mobil_curves.png?raw=true)

Due to time limitations of the project, we were not able to train the MobileNet model with more epochs. However, considering the above learning curves we can observe some highly probable possibility of improvement if we allowed more epochs for training.

# Results

The following are the results of evaluating the F1 score using the union of the detection and classification models:

| Model            	| F1 Score 	|
|------------------	|----------	|
| Convnet baseline 	| 0.7994   	|
| ResNet18         	| 0.5647   	|
| MobileNet        	| 0.7756    |

Trained models are available for download at the following links:

- [CenterNet](https://drive.google.com/file/d/12u0bbY1u3Odaijab4SLtUmQMPx_ZOWbD/view)
- [ConvBaseline](https://drive.google.com/file/d/1sNq5rJF6cakcSII8QRg6o7kXQMUb6V82/view?usp=sharing)
- [ResNet18](https://drive.google.com/file/d/10oi6K265-fArTQwilK9UdA3VdK3fMtio/view)
- [MobilNetV3](https://drive.google.com/file/d/1wHlt1tV9V_viSNwETN10-VihFP4FE52d/view)

# References

- [1] A. Lamb, *How Machine Learning Can Help Unlock the World of Ancient Japan*, The Gradient https://thegradient.pub/machine-learning-ancient-japan/ (2019), last accessed 15.03.2021

- [2] *Kuzushiji Recognition*, URL: https://www.kaggle.com/c/kuzushiji-recognition/data, last accessed 18.04.2021

- [3] Zhou et al., [*Objects as Points*](https://arxiv.org/abs/1904.07850), Computer Vision and Pattern Recognition (2019)

- [4] He et al., [*Deep residual learning for image recognition*](https://arxiv.org/abs/1512.03385), Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2016)

- [5] Howard et al., [*Searching for MobileNetV3*](https://arxiv.org/abs/1905.02244), IEEE/CVF International Conference on Computer Vision (ICCV) (2019)

- [6] Cui et al., [*Class-Balanced Loss Based on Effective Number of Samples*](https://arxiv.org/abs/1901.05555), Computer Vision and Pattern Recognition (2019)
