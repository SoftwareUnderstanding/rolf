### Description
 built a plant disease diagnosis system on Android, by implementing a deep convolutional neural network with Tensorflow to detect disease from various plant leave images. 

Generally, due to the size limitation of the dataset, we adopt the transefer learning in this system. Specifically, we retrain the MobileNets [[1]](https://arxiv.org/pdf/1704.04861.pdf), which is first trained on ImageNet dataset, on the plant disease datasets. Finally, we port the trained model to Android.

### Configurations
* [Tensorflow 1.3](https://www.tensorflow.org)
* [Android Studio 3.0 Preview](https://developer.android.com/studio/preview/index.html)
* Nexus 6 (Android 7.1)
* If you have want know more about Tensorflow on Android, please find more details at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android

### Dataset
The dataset [[2]](https://arxiv.org/pdf/1604.03169.pdf) contains 54,306 images of plant leaves, which have a spread of 38 class labels assigned to them. Each class label is a crop-disease pair, and we make an attempt to predict the crop-disease pair given just the image of the plant leaf.

The dataset is available at https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge/dataset_files.

### Reference
[1] Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).

[2] Mohanty, Sharada P., David P. Hughes, and Marcel Salathé. "Using deep learning for image-based plant disease detection." Frontiers in plant science 7 (2016).

### RESULTS
major screenshots
