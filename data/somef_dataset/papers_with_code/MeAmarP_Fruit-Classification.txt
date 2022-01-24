# Fruit-Classification: Work-In-Progress
Fruit Classification/Identififcation using TensorFlow-Keras on Fruits 360 dataset

## Understand Dataset:
![Understanding Dataset][EDA_Img]

[EDA_Img]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/EDA_images_v22.png

### Step 1 - EDA:

__Method/Code Snippet:__
```python
#get path to root dir
base_dir_path = os.getcwd()

#build path to train dir
train_dir_path = os.path.join(base_dir_path,'train')

#build path to test dir
test_dir_path = os.path.join(base_dir_path,'test')

readData(base_dir_path)
```
__Console Output:__
```console
Total Number of Classes in train DataSet:  95
Total Number of Classes in test DataSet:  95
Total Number of train samples:  48905
Total Number of test samples: 16421
```
__Method/Code Snippet:__
```python
understandData(base_dir_path,'train')
```
__Console Output:__
```console
CLASS NAME          NUMBER OF IMAGES
Apple Braeburn      492
Apple Golden 1      492
Apple Golden 2      492
Apple Golden 3      481
Apple Granny Smith  492
.
.
.
```

## Build Model and Train Dataset:

### Approch:
+ I used MobileNetV2 architecutre, pre-trained on ImageNet dataset for feature extraction.
+ Next I use these features and ran through a new classifier, which is trained from scratch.
+ As stated in my Favourite Book: __Deep Learning with Python__. 
We took convolutional base(conv_base) of MobileNetV2, ran new data through it and trained a new classifier on top of
the output.
+ So basically, I extended the conv_base by adding Dense layer followed by DropOut layer, and running 
whole network on input data with data augmentation. 
+ Well!! this is computationally bit expensive, but meh!! I have enough processing power.
+ Important Thing, I freeze the convolutional base so as to avoid updating their weights.

### Step 2 - Compiling Model:
__Method/Code Snippet:__
```python
#Get list of All classes
AllClassNames = getAllClassNames(train_dir_path)
num_of_classes = len(AllClassNames)

#build dict of clas_id and classname
DictOfClasses = {i : AllClassNames[i] for i in range(0, len(AllClassNames))}

#Compile classification model
classifyModel=compileClassifyModel(num_of_classes)
```
__Console Output:__
```console
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Model) (None, 1280)              2257984   
_________________________________________________________________
flatten_1 (Flatten)          (None, 1280)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               655872    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 95)                48735     
=================================================================
Total params: 2,962,591
Trainable params: 704,607
Non-trainable params: 2,257,984
_________________________________________________________________
```

### Step 3 - Training compiled Model:
__Method/Code Snippet:__
```python
#Start training model on train dataset
trainingHistory,trainedModel_filename = trainClassifyModel(classifyModel)

#Plot the training results
plotTrainResults(trainingHistory)
```
### Training Results:
**Epcohs:20**

![train_valid_acc][plot_acc]

[plot_acc]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_acc_16JUL_20epochs.png

![train_valid_loss][plot_loss]

[plot_loss]: https://github.com/MeAmarP/Fruit-Classification/blob/master/results/train_valid_Loss_16JUL_20epochs.png


### Step 4 - Prediction:
__Method/Code Snippet:__
```python
#path to test image
ImagePath = 'test/Banana Red/99_100.jpg'

#path to trained-saved model
path_trained_model = os.path.abspath(trainedModel_filename)

#load trained model
trainedModel = getTrainedModel(path_trained_model)

#perform predictions
AllProbs = predictFruitClass(ImagePath,trainedModel,DictOfClasses)
```
__Console Output:__
```console
Banana
```

## Issues and Challenges:
+ Need more diverse data for each fruit class.
+ It is really hard for model to infer the type of fruit, this may be due to closer properties(shape,color etc) of the object.
I mean, it is easier for model to recognise Banana compared to other fruit class.
+ For Example, model predicts Grape White as Guava. __Refer Above grid image__. This has been observed with several
other fruit classes. 

### TODO:
- [ ] Test with more epochs.
- [ ] Test with ResNet, InveptionV3,Xception models
- [ ] Add method to print Top-K predicted categories/classes.
- [ ] Add method to Evaluate prediction accuracy and loss on whole test dataset.


## Refrences:
+ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
 <https://arxiv.org/abs/1704.04861>
+ <https://keras.io/applications/#resnet>
+ Deep Learning with Python, François Chollet.