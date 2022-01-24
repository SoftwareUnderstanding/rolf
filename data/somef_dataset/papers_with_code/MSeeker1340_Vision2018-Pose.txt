# Vision2018-Pose
Class project repository for CSCI-GA.3033-012: Vision Meets Machine Learning (Fall 2018)

We explore and experiment the methods and processes of real-time 2D multiperson pose estimation described in the paper: https://arxiv.org/pdf/1611.08050.pdf

Previous results and references:
1. Original version: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
2. OpenPose (C++ version): https://github.com/CMU-Perceptual-Computing-Lab/openpose
3. OpenPose Plus (Latest version using TensorFlow, OpenPose and TensorRT): https://github.com/tensorlayer/openpose-plus

## Download and Unzip Data
get_data.sh

## Preprocessing 
src/preprocessing.py

### API
Load preprocessed data from its function: 
```python
X, Y = load_data(data_dir, data_type, image_shape=image_shape, sigma=8.0, num_input=None, verbose=False, image_ids = None)
```
X —— Image

Y —— Confidence map

### What it does
1. Load local data using COCO protocol, including images and annotations.
2. Resize each image to X, an array of images (s1,s2,3), where (s1,s2) is the image_shape parameter (default s1=s2=224). 
We require that s is divisible by 8.
3. COCO annotations label the body joints(src/config.py KEYPOINTS) already, and we use these points to build confidence map of each image and store in Y. Y is an array of (s/8, s/8, NUM_KEYPOINTS).
The confidence map is calculated using a Gaussian kernel and take max

## Plot and Examine Confidence Maps
notebooks/Preprocessing.ipynb
src/plot.py

### API
```python
plot_on_img(x, y, f)
```
f —— The index of a key point

### What it does
Plot confidence map Y on image X

## Pretrain X using VGG19 model and get the first 10 layers Z
src/vgg.py

### API
```python
Z = prestage(X.copy())
```

## Train
notebooks/Train.ipynb
src/model.py

### API
```python
stage1.fit(Z, Y, epochs=15, validation_split=0.2)
stage1.save_weights("../dataset/models/stage1_weights.h5")
```

### What it does
Pass Z through the model using Adam optimizer. For each layer, evaluate trained Z and Y's mean square error to check convergence.

Model specification:
1. Normalize features of each channel
2. Reduce the number of feature maps
3. Train the first branch (confidence map) of stage 1

At last, save the normalization weights for prediction

## Test
notebooks/Test.ipynb
src/model.py
```python
stage1.load_weights("../dataset/models/stage1_weights.h5")
Y_predict = stage1.predict(Z)
```
src/vgg.py
```python
Z = prestage(X.copy())
```
src/plot.py
```python
plot_on_img(x, y, f)
```

### Test procedure
1. Load data X,Y from COCO (just like training)
2. Load weights
3. Preprocess X using VGG19 and store it in Z
4. Predict using trained model
5. Plot results
