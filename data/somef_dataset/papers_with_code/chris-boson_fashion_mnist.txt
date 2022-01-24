# FashionMNIST

## Installation
* Create and activate new virtual env with python3
* `pip install -r requirements.txt`

## Usage
* Set parameters (see `main.py`)
* `./run_train.sh`

## Experiments

### Architectures
#### CNN
2 Conv Blocks (Conv, ReLU, max pool) + 2 FC. This was meant as simple baseline, although given its very small footprint accuracy is still competetive.
#### CNNwithBN
5 Conv Blocks + 1 FC (Conv, ReLU6, BN with max pooling in later layers). Very standard architecture these days, ReLU6 improves resilience when quantizing the network, BN improves training stability. In a second experiment separable convolutions are used as drop in replacement for regular Conv2d. This reduces the number of parameters by a factor of 8 and number of ops by 5, while reducing accuracy by 2%.

#### MobileNetV2
Architecture optimized for mobile devices and is commonly used as backend for image classification. While number of parameters is higher, the number of operations is greatly reduced compared to naive `CNNwithBN Seperable`. Comparing it to the `CNN` baseline it may still be overkill given the simplicity of this task (only 1 channel 28x28 images).
https://arxiv.org/abs/1801.04381

#### MobileNetV3
Optimized version of `MobileNetV2`. While smaller and more compute efficient, accuracy seems to fall short significantly.
https://arxiv.org/abs/1905.02244

### Data Augmentation
We use horizontal flips as a regularization technique. Here we don't do a detailed breakdown on its effect on accuracy, but experiments showed no significant boost.

### Results

| Architecture        | Accuracy | GFlops  | MParams  |
|---------------------|----------|---------|----------|
| CNN                 | 0.899    | 0.00027 | 0.032998 |
| CNNwithBN           | 0.934    | 0.17586 | 1.575114 |
| CNNwithBN Separable | 0.914    | 0.03603 | 0.187028 |
| MobileNetV2         | 0.924    | 0.00561 | 2.236106 |
| MobileNetV3         | 0.907    | 0.00346 | 1.66989  |