# Flower-Classifaction-with-Fine-Tuned-Mobilenet

## [Datasets for Training]
Download  the flower datasets from below kaggle link.

https://www.kaggle.com/alxmamaev/flowers-recognition

Flower dataset contains 'daisy', 'dandelion' , 'rose', 'sunflower' and 'tulip'.

lets rearrange the datasets as mentioned below.

* Datasets:
	* Train
	* Validation
	* Test

For experiment purpose lets Train the model with 5X500 flowers.

## [Dataset preprocessing before traing]
For image augmentation used ImageDataGenerator class from keras (from tensorflow.keras.preprocessing.image import ImageDataGenerator)

## [Fine Tunning mobilenet]
Lets call mobilnet model using tf.keras.applications.mobilenet.MobileNet()
Note: make sure network connection is available.
we will not use the last five layers of the original model and specify the 5 units in Dense layer as we have 5 classes.

x = model.layers[-6].output
output = Dense(units=5, activation='softmax')(x)

for layer in model.layers[:-13]:
    layer.trainable = False

we can  play with parameters/Hyper parameters.

## [Traing the Fine tuned Model]

self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

self.model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

self.model.fit(x=self.train_batches,steps_per_epoch=len(self.train_batches),validation_data=self.valid_batches,validation_steps=len(self.valid_batches),epochs=self.epochs,verbose=2, callbacks=[tensorboard_callback])

```
Epoch 1/30
251/251 - 133s - loss: 0.5820 - accuracy: 0.7888 - val_loss: 0.1084 - val_accuracy: 0.9800
Epoch 2/30
251/251 - 15s - loss: 0.2200 - accuracy: 0.9275 - val_loss: 0.1071 - val_accuracy: 0.9800
Epoch 3/30
251/251 - 15s - loss: 0.1373 - accuracy: 0.9570 - val_loss: 0.1931 - val_accuracy: 0.9000
Epoch 4/30
251/251 - 15s - loss: 0.0788 - accuracy: 0.9809 - val_loss: 0.0578 - val_accuracy: 0.9800
Epoch 5/30
251/251 - 15s - loss: 0.0707 - accuracy: 0.9801 - val_loss: 0.0785 - val_accuracy: 1.0000
Epoch 6/30
251/251 - 15s - loss: 0.0469 - accuracy: 0.9916 - val_loss: 0.1873 - val_accuracy: 0.9400
Epoch 7/30
251/251 - 15s - loss: 0.0309 - accuracy: 0.9964 - val_loss: 0.0555 - val_accuracy: 0.9800
Epoch 8/30
251/251 - 15s - loss: 0.0341 - accuracy: 0.9936 - val_loss: 0.0166 - val_accuracy: 1.0000
Epoch 9/30
251/251 - 15s - loss: 0.0236 - accuracy: 0.9964 - val_loss: 0.2976 - val_accuracy: 0.9400
Epoch 10/30
251/251 - 15s - loss: 0.0196 - accuracy: 0.9984 - val_loss: 0.1444 - val_accuracy: 0.9400
Epoch 11/30
251/251 - 15s - loss: 0.0215 - accuracy: 0.9960 - val_loss: 0.0302 - val_accuracy: 0.9800
Epoch 12/30
251/251 - 16s - loss: 0.0195 - accuracy: 0.9964 - val_loss: 0.0639 - val_accuracy: 0.9800
Epoch 13/30
251/251 - 16s - loss: 0.0155 - accuracy: 0.9980 - val_loss: 0.0495 - val_accuracy: 0.9800
Epoch 14/30
251/251 - 16s - loss: 0.0120 - accuracy: 0.9992 - val_loss: 0.0641 - val_accuracy: 0.9800
Epoch 15/30
251/251 - 15s - loss: 0.0115 - accuracy: 0.9984 - val_loss: 0.0345 - val_accuracy: 0.9800
Epoch 16/30
251/251 - 15s - loss: 0.0138 - accuracy: 0.9968 - val_loss: 0.0780 - val_accuracy: 0.9800
Epoch 17/30
251/251 - 15s - loss: 0.0102 - accuracy: 0.9992 - val_loss: 0.0392 - val_accuracy: 0.9800
Epoch 18/30
251/251 - 15s - loss: 0.0207 - accuracy: 0.9932 - val_loss: 0.0204 - val_accuracy: 1.0000
Epoch 19/30
251/251 - 15s - loss: 0.0283 - accuracy: 0.9920 - val_loss: 0.0494 - val_accuracy: 0.9600
Epoch 20/30
251/251 - 15s - loss: 0.0119 - accuracy: 0.9972 - val_loss: 0.1433 - val_accuracy: 0.9600
Epoch 21/30
251/251 - 15s - loss: 0.0094 - accuracy: 0.9980 - val_loss: 0.0261 - val_accuracy: 1.0000
Epoch 22/30
251/251 - 15s - loss: 0.0108 - accuracy: 0.9976 - val_loss: 0.0807 - val_accuracy: 0.9600
Epoch 23/30
251/251 - 15s - loss: 0.0063 - accuracy: 0.9988 - val_loss: 0.1585 - val_accuracy: 0.9400
Epoch 24/30
251/251 - 15s - loss: 0.0062 - accuracy: 0.9988 - val_loss: 0.0777 - val_accuracy: 0.9800
Epoch 25/30
251/251 - 15s - loss: 0.0077 - accuracy: 0.9976 - val_loss: 0.0088 - val_accuracy: 1.0000
Epoch 26/30
251/251 - 15s - loss: 0.0067 - accuracy: 0.9988 - val_loss: 0.0515 - val_accuracy: 0.9800
Epoch 27/30
251/251 - 15s - loss: 0.0088 - accuracy: 0.9980 - val_loss: 0.1085 - val_accuracy: 0.9600
Epoch 28/30
251/251 - 15s - loss: 0.0122 - accuracy: 0.9972 - val_loss: 0.1371 - val_accuracy: 0.9600
Epoch 29/30
251/251 - 15s - loss: 0.0126 - accuracy: 0.9960 - val_loss: 0.0393 - val_accuracy: 0.9800
Epoch 30/30
251/251 - 15s - loss: 0.0189 - accuracy: 0.9924 - val_loss: 0.1077 - val_accuracy: 0.9800
<tensorflow.python.keras.callbacks.History at 0x7fca17885e50>
```

## [Tensorboard]
![image](https://user-images.githubusercontent.com/76731781/121777411-87068980-cbaf-11eb-8ede-91a69775d902.png)


## [Predictions]
```
No.:0  - Label:daisy     - Predicted:daisy
No.:1  - Label:daisy     - Predicted:daisy
No.:2  - Label:daisy     - Predicted:daisy
No.:3  - Label:daisy     - Predicted:daisy
No.:4  - Label:daisy     - Predicted:daisy
No.:5  - Label:dandelion - Predicted:dandelion
No.:6  - Label:dandelion - Predicted:dandelion
No.:7  - Label:dandelion - Predicted:dandelion
No.:8  - Label:dandelion - Predicted:dandelion
No.:9  - Label:dandelion - Predicted:dandelion
No.:10 - Label:rose      - Predicted:rose
No.:11 - Label:rose      - Predicted:rose
No.:12 - Label:rose      - Predicted:rose
No.:13 - Label:rose      - Predicted:rose
No.:14 - Label:rose      - Predicted:rose
No.:15 - Label:sunflower - Predicted:sunflower
No.:16 - Label:sunflower - Predicted:sunflower
No.:17 - Label:sunflower - Predicted:sunflower
No.:18 - Label:sunflower - Predicted:sunflower
No.:19 - Label:sunflower - Predicted:sunflower
No.:20 - Label:tulip     - Predicted:tulip
No.:21 - Label:tulip     - Predicted:tulip
No.:22 - Label:tulip     - Predicted:tulip
No.:23 - Label:tulip     - Predicted:tulip
No.:24 - Label:tulip     - Predicted:tulip
```
## [References for further reading]
https://arxiv.org/pdf/1704.04861.pdf

https://deeplizard.com/learn/video/Zrt76AIbeh4

https://github.com/Machine-Learning-Tokyo/CNN-Architectures/tree/master/Implementations/MobileNet
