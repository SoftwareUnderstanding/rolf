# AI Project

Dataset: https://drive.google.com/uc?id=0B_lM0116PvbEOVR4MVBmdXMwX3M&export=download

## Model Iterations

### Summary

| Version | Validation Accuracy | Model Size | Training Time |
| ------- | ------------------- | ---------- | ------------- |
| 1       | 71.88%              | 219 MB     | 3 Hours       |
| 2       | 39.53%              | 16 MB      | 7.6 Minutes   |
| 3       | 48.59%              | 16 MB      | 14.9 Minutes  |
| 4       | 67.19%              | 7 MB       | 10 Minutes    |

### Version 1

``` txt
Train Set: 6400
Validation Set: 640
Batch size: 64
Epochs: 10
Training Time: 10387s

Number of Classes: 10
Class Labels: {0: 'samgupsal', 1: 'bulgogi', 2: 'ojingeo_bokkeum', 3: 'dakbokkeumtang', 4: 'galchijorim', 5: 'jeyuk_bokkeum', 6: 'ramen', 7: 'bibimbap', 8: 'galbijjim', 9: 'kimchi'}

Input Shape: 224 x 224 x 3

Image Augmentation: None

Best Validation Accuracy: 72.34%

Base Model: Inception Resnet V2
Inception Resnet Paper: https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14806
TF Hub: https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4

Model Size: 219328 KB
```


### Version 2

In this version, I switched to a smaller model as the Inception Resnet is to large to be deployed on a FREE Heroku server. The validation accuracy in this model is significantly lower but it is faster to train.

``` txt
Train Set: 6400
Validation Set: 640
Batch size: 32
Epochs: 1
Training Time: 458s

Number of Classes: 10
Class Labels: {0: 'samgupsal', 1: 'bulgogi', 2: 'ojingeo_bokkeum', 3: 'dakbokkeumtang', 4: 'galchijorim', 5: 'jeyuk_bokkeum', 6: 'ramen', 7: 'bibimbap', 8: 'galbijjim', 9: 'kimchi'}

Input Shape: 224 x 224 x 3

Image Augmentation: None

Best Validation Accuracy: 39.53%

Base Model: MobileNet V2
MobilenetV2 Paper: https://arxiv.org/abs/1801.04381

Model size: 16632 KB
```


### Version 3

In this version I used image augmentation to increase the training dataset size. The validation accuracy improved as a result.

``` txt
Train Set: 6400
Validation Set: 640
Batch size: 32
Epochs: 2
Training Time: 891s

Number of Classes: 10
Class Labels: {0: 'samgupsal', 1: 'bulgogi', 2: 'ojingeo_bokkeum', 3: 'dakbokkeumtang', 4: 'galchijorim', 5: 'jeyuk_bokkeum', 6: 'ramen', 7: 'bibimbap', 8: 'galbijjim', 9: 'kimchi'}

Input Shape: 224 x 224 x 3

Image Augmentation: ImageDataGenerator(rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    brightness_range=[0.7, 1.3],
                    shear_range=0.2,
                    zoom_range=0.5,
                    horizontal_flip=True,
                    vertical_flip = True,
                    rescale=1/255)

Best Validation Accuracy: 48.59%

Base Model: MobileNet V2
MobilenetV2 paper: https://arxiv.org/abs/1801.04381

Model size: 16629 KB
```

### Version 4

In this version, I try to use an updated version of mobilenetv2 on tfhub, it was able to train the model more efficiently and accurately than the previous versions.

I did not modify this model first as I had trouble deploying a model that was built on tfhub on Heroku.

``` txt
Train Set: 6400
Validation Set: 640
Batch size: 32
Epochs: 10 (early stop at 4)
Training Time: 665s

Number of Classes: 10
Class Labels: {0: 'samgupsal', 1: 'bulgogi', 2: 'ojingeo_bokkeum', 3: 'dakbokkeumtang', 4: 'galchijorim', 5: 'jeyuk_bokkeum', 6: 'ramen', 7: 'bibimbap', 8: 'galbijjim', 9: 'kimchi'}

Input Shape: 224 x 224 x 3

Image Augmentation: ImageDataGenerator(rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    brightness_range=[0.7, 1.3],
                    shear_range=0.2,
                    zoom_range=0.5,
                    horizontal_flip=True,
                    vertical_flip = True,
                    rescale=1/255)

Validation Accuracy: 67.19%

Base Model: MobileNet V2
MobilenetV2 paper: https://arxiv.org/abs/1801.04381
TF Hub: https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4

Model size: 6963 KB
```

### Model Comparison

The best validation accuracy out of all the versions of mobileNetV2 models I have trained is 67.19%, whereas the original mobileNetV2 trained on ImageNet has an accuracy of 74.7%. This could be due to a small training set.

The best validation accuracy for Inception Resnet V2 is 72.34% while the validation accuracy of Inception Resnet V2 is 80%.
