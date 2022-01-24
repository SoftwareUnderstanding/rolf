
# The Identification Game
### The Data
This project is an image classification problem consisting of Natural Images. You are provided a dataset consisting of images and associated labels. Most of the images have 3 channels for colour, RGB, meaning they are 64x64x3 arrays. Each image belongs to exactly one out of 200 categories. The dataset contains 100k images (500 from each category). The test set have no labels. Further validation/test splits are needed for the training process.

Some images from the train dataset from different classes:
<p align ="middle">
    <img src="https://github.com/acse-2019/acse4-4-logistic/blob/master/train_images_preview/n01443537_10.JPEG" width="100"/><img src="https://github.com/acse-2019/acse4-4-logistic/blob/master/train_images_preview/n02791270_2.JPEG" width="100"/><img src="https://github.com/acse-2019/acse4-4-logistic/blob/master/train_images_preview/n02814533_2.JPEG" width="100"/><img src="https://github.com/acse-2019/acse4-4-logistic/blob/master/train_images_preview/n09193705_7.JPEG" width="100"/>
</p>

### The Task
Train a classifier that can take as input an image that is similar to the ones provided in the dataset, and output a class corresponding to this image.

All information about the identification game is available here: https://www.kaggle.com/c/acse-miniproject/overview.

## User guide
We implemented multiple models to classify 100,000 images into 200 classes (500 in each class). To train the models, simply run the Jupyter notebook [identification_game.ipynb](identification_game.ipynb).

The trained models can be saved as files with **.pth** as extension:
>torch.save(model_name, save_path)

when you want to continue training, the trained models can be easily loaded with:
>model = torch.load('./model_name.pth')

## Implementation
The models which have top two Kaggle scores are Inception_v3 and ResNet 101. The model architecture and experiment results can be found in the presentation slide for the project [acse4-4-logistic.pptx](acse4-4-logistic.pptx).

## References
### About network architecture and implementation
1. ResNet 101: https://arxiv.org/pdf/1512.03385.pdf
2. Inception v3: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf
3. Wide ResNet: https://arxiv.org/pdf/1611.05431.pdf

### About training techniques
1. Data augmentation: https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
2. Transfer learning and other tricks: https://www.oreilly.com/library/view/programming-pytorch-for/9781492045342/ch04.html
3. Tricks for training deep neural networks: https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8
