# Microsoft-Cats-and-Dogs-Image-Classification
## Fine Tune ResNet50 + Random Erasing Augmentation reach 99% Accuracy
Random Erasing is a data augmentation method for training the convolutional neural network. In training, Random Erasing randomly
selects a rectangle region in an image and erases its pixels with random values. In this process, training images 
with various levels of occlusion are generated, which reduces the risk of over-fitting and makes the model robust
to occlusion.

---

## Install Package
```sh
pip install -r requirements.txt
```

---

## Data Visualization

- Rotation and Shift by Data Generator
```python
datagen = ImageDataGenerator(rotation_range = 40,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             channel_shift_range = 10,
                             horizontal_flip = True,
                             fill_mode = 'nearest')
```
<img src="./readme_img/1.PNG" width="250"> <img src="./readme_img/2.PNG" width="250"> <img src="./readme_img/3.PNG" width="250">
<img src="./readme_img/4.PNG" width="250"> <img src="./readme_img/5.PNG" width="250">

---

- Add Random Erasing Method to Train Data

<img src="./readme_img/e_1.PNG" width="250"> <img src="./readme_img/e_2.PNG" width="250"> <img src="./readme_img/e_3.PNG" width="250">
<img src="./readme_img/e_4.PNG" width="250"> <img src="./readme_img/e_5.PNG" width="250">

---

## Fine Tune ResNet50
- Get imagenet weights
```python
res50 = ResNet50(input_shape = img_size,
                 weights = "imagenet",
                 include_top = False)
x = GlobalAveragePooling2D()(res50.output)
output = Dense(2, activation = 'softmax')(x)
model = Model(inputs = res50.input, outputs = output, name = "resnet50_transfer")
```
- Freeze some layers
```python
for ind,layer in enumerate(model.layers):
    if layer.name == "conv2_block3_3_conv":
        freeze_layer = ind
for layer in model.layers[:freeze_layer]:
    layer.trainable = False
for layer in model.layers[freeze_layer:]:
    layer.trainable = True
```
- Learning rate scheduler & Early Stopping
```python
callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=6, verbose=1, min_lr = 1e-7)
opt = Adam(learning_rate = 1e-5)
```

---

## Result
- Test Data Accuracy

<img src="./readme_img/r_1.PNG" width="650">

---

- Training Phase Plot

<img src="./readme_img/r_2.PNG" width="350">
<img src="./readme_img/r_3.PNG" width="350">

---
- Classification Report

<img src="./readme_img/r_4.PNG" width="550">

---

## Reference
https://arxiv.org/pdf/1708.04896.pdf
