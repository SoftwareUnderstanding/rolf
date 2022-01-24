# Pneumonia-detection-Dense-Conv-Net
Pneumonia Detection using Chest XRay Classification

## About:
This is a pytorch implementation of classification of chest X-rays as infected with pneumonia or not. This is examined by radiologists manually for air cavities and lumps. The model also outputs a heat map indicating the areas in the xray dominant in leading to the prediction.
The paper [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning (https://arxiv.org/abs/1711.05225)] was published in Dec 2017 by Pranav Rajpurkar and Jeremy Irvin.

Citations:
https://arxiv.org/abs/1711.05225

## Broader look out:
The 121 layered convolutional neural network has acheived the state of the art detection of pneumonia over manual detection by human radiologist, compared by F1 scores. It uses a dataset of 100000 Chest X-rays 14 dataset annotated by four radiologists.

### Examples
![ChestXray input](https://github.com/MicroprocessorX069/Pneumonia-detection-Dense-Conv-Net/blob/master/documentation/image%20res/chestxray.PNG)
![ChestXray cam output](https://github.com/MicroprocessorX069/Pneumonia-detection-Dense-Conv-Net/blob/master/documentation/image%20res/chestxraycam.PNG)

### Architecture


### Prerequisites
### Usage

#### Requirements
- Python 3.5+
- PyTorch 0.3.0
 
#### Directory structure
The directory structure is followed as 
```
.
├── ...
├── version_no                    # Version of different models and training process
│   ├── model          # saved model checkpoint files
│   ├── report         # reporting of final training, validation loss and other metrics
│   └── output          # Output directory
│       └── epoch                    # Storing the training epoch images
├── data                    # Dataset of images (Optional)
├── res                # Resources directory
│    └── Helvetica                    # Font file to generate paired images for training (optional) 
└── ...
```

#### Train/ test
1. Clone the repository
```
$ git clone https://github.com/MicroprocessorX069/Pneumonia-detection-Dense-Conv-Net.git
$ cd Pneumonia-detection-Dense-Conv-Net
```
2. Install the libraries
```
pip3 install requirements.txt
```
3. Train
(i) Download data
```

```
(ii)Train the model
```
$ python python train.py
```
4.Test
```
$ python app.py
```
![homepage](https://github.com/MicroprocessorX069/Pneumonia-detection-Dense-Conv-Net/blob/master/documentation/image%20res/homepage.PNG)

Opening the localhost URL in the browser.

Upload any chest Xray image and click to get the results.

#### Using a pretrained model weights
Download the model weights as .ckpt file in "./model/" and hit the same commands to train and test with the correct root directory.

## Results
![Training gif]()

## Implementation details
- [Theoritical details](docs/CONTRIBUTING.md)
- [Modules](docs/CONTRIBUTING.md)
- [Data](docs/CONTRIBUTING.md)
- [Architecture](docs/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Class activation mappings](documentation/cam.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)

## Issues
- [How to handle overfitting](documentation/regularization.md)
- [Modules](docs/CONTRIBUTING.md)
- [Data](docs/CONTRIBUTING.md)
- [Architecture](docs/CONTRIBUTING.md)
- [Code structure](docs/CONTRIBUTING.md)
- [Class activation mappings](documentation/cam.md)
- [Distributed training](docs/CONTRIBUTING.md)
- [Docker](docs/CONTRIBUTING.md)

## Related projects
## Acknowledgements




