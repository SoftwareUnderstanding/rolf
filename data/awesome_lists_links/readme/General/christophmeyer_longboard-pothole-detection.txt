# Longboard Pothole Detection

In this project, a model is trained to detect irregularities on the surface in front of a longboard to warn the rider by activating a piezo buzzer. The repository contains all code necessary to collect training data, train and run a model on an ESP32 CAM using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). It also contains the tflite version of a model that was trained on data that I collected and labeled. 


![](media/longboard-front-view.jpg)    ![](media/ride_model_perspective.gif) ![](media/longboard-demo.gif)

For a demo video version with sound (to hear the warning buzzer) take look [here](https://www.youtube.com/watch?v=RgqTGeaQ5AA).

### Why?

The idea for this project is to warn the rider to pay attention to the surface in front of them to avoid falling, but mostly the project is an excuse to play with the ESP32 CAM and TF Lite Micro :smiley:.

## Table of contents

- [Getting started](#getting-started)
- [Data Capture](#data-capture)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Conversion](#model-training-and-conversion)
- [Inference](#inference)

## Getting started

To get started, you will need a few pieces of hardware. Everything can be put together with jumper cables without soldering.

### List of parts

| Part      | Approx. Price       | Comment   |
| --------- | ------------------- | --------- |
| ESP32 CAM |       10 EUR        | Make sure to get the version with 4MB PSRAM. |
| UART -> micro USB adapter P2303    | 4 EUR |  | 
| OV2640 camera      | 1.5 EUR | Depending on your mounting situation, make sure to choose one with a long enough cable (I used 75mm). Since the cables easily fail if you bend them too much, you might want to order a spare or two.  | 
| KY-012 active piezo buzzer  | 2.5 EUR |  | 
| At least 10 f/f jumper cables     | 3 EUR | Make sure they are long enough (I use 250mm ones) so you can arrange the parts on your longboard as you wish.   | 
| USB powerbank | 10 EUR | Powerbank to power the ESP32 CAM on the longboard.  
| USB-A -> micro USB cable  | 3 EUR | For connecting the UART adapter to the computer or the powerbank. | 
| 4GB microSD card  | 3 EUR | Only needed if you want to capture your own data (which I highly recommend). | 
| MPU6050 IMU | 3EUR | (Optional) if you also want to capture IMU data. Unless you want to solder, get one with header pins.

For a total of 40EUR (~50USD) you can get all the electronics parts for this project. Apart from those parts, your longboard and a computer with sdcard reader, you will need some means to mount the parts on your longboard. Here is how I did that:

For the ESP32 CAM and the camera, I came up with a contraption made of wood and tape:  

![](media/case-prototype.jpg) ![](media/esp32-case-side-view.jpg)

For the UART adapter and buzzer I found it to be sufficient to tape them to the longboard, in case of the UART adapter I also used some foam for protection:

![](media/uart-buzzer-on-longboard.jpg) ![](media/uart-covered-in-foam.jpg) 

For the IMU, I use the two screw holes on the MPU6050 board to screw it to a small piece of wood, which I then taped to the longboard with a strong double-sided tape. For protection, I then covered it with a piece of foam. 

![](media/imu-mounted-on-wood.jpg) ![](media/imu-taped-to-longboard.jpg)

The following image shows the longboard in inference configuration (with buzzer, no IMU). The powerbank is taped to the board at the back:

![](media/inference-config-longboard.jpg)

### Setup 

First clone this repo:

```
git clone https://github.com/christophmeyer/pothole-detection-longboard.git
```

The project is divided into three subprojects:

`./esp32_data_collection`: C++ code to build the esp32 binary for data collection, i.e. for saving grayscale pictures and (optional) imu data to the sdcard. 

`./pothole_model`: Python code for preparing the collected data for training, training the model, and converting/quantizing the model such that it can be compiled into the inference binary.

`./esp32_inference`: C++ code to build the esp32 binary for running inference and driving the piezo buzzer.

Both, the data collection and inference subprojects use the esp-idf build system, which you can set up by following the steps [here](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/). 

Next, one needs to fetch the camera drivers for both esp32 subprojects:

```
git clone https://github.com/espressif/esp32-camera esp32_data_collection/components/esp32-camera
cd ./esp32_data_collection/components/esp32-camera
git checkout 722497cb19383cd4ee6b5d57bb73148b5af41b24

git clone https://github.com/espressif/esp32-camera esp32_inference/components/esp32-camera
cd ./esp32_inference/components/esp32-camera
git checkout 722497cb19383cd4ee6b5d57bb73148b5af41b24
```

For the IMU, one needs the I2C and MPU drivers:

```
git clone https://github.com/natanaeljr/esp32-I2Cbus.git esp32_data_collection/components/I2Cbus
cd ./esp32_data_collection/components/I2Cbus
git checkout c792c7b5141772f805766a289b86819664894b23

git clone https://github.com/natanaeljr/esp32-MPU-driver.git esp32_data_collection/components/MPUdriver
cd ./esp32_data_collection/components/MPUdriver
git checkout c82b00502eb4c101a3f6b8134cd9b4a13f88e016
```

The above commits of the driver repos are the ones that I used. Using newer ones might be fine as well, in particular in the case of the MPU and I2C drivers. However, for the camera driver, I encountered a bug with grayscale images that was introduced after the commit mentioned above. So check for yourself if you want to use more recent drivers.

### How to connect things

The pins of the ESP32 CAM need to be connected to the other components depending on the scenario (flashing a new binary/collecting data/running inference).

0. General remarks:
- The OV2640 camera can be connected at all times, just plug the ribbon cable into its slot on the front of the ESP32 CAM. 
- We use the UART adapter as power supply both when connected to the computer via USB and on the longboard. To set the UART adapters VCCIO output voltage to 5V (which I found to work very reliably), you need to set the small yellow jumper to 5V. Then the power lines can be connected:

```
VCCIO (UART Adapter) <-> 5V (ESP32 CAM)
GND (UART Adapter) <-> GND (ESP32 CAM)
```

For the serial connection, two more jumper cables are needed:

```
TXD (UART Adapter) <-> GPIO 3 (ESP32 CAM)
RXD (UART Adapter) <-> GPIO 1 (ESP32 CAM)
```


1. Flashing a new binary:

To put the ESP32 CAM into flashing mode, GPIO 0 needs to be grounded (i.e. connected with a jumper to GND).

2. Collecting data:

If you are not interested in capturing additional IMU data with the MPU6050, you only need to connect the power lines as described above.

For collecting IMU data, you need to connect the MPU6050 as follows:

```
VCC (MPU6050) <-> VCC (ESP32 CAM)
GND (MPU6050) <-> GND (ESP32 CAM)
SCL (MPU6050) <-> GPIO 1 (ESP32 CAM)
SDA (MPU6050) <-> GPIO 3 (ESP32 CAM)
```

This of course means that you cannot use the serial connection in this configuration. 

3. Inference:

Apart from connecting the power lines as described above, the KY-012 active piezo buzzer needs to be connected as follows:

```
GND (KY-012) <-> GND (ESP32 CAM)
S (KY-012) <-> GPIO 16 (ESP32 CAM)
```

### Building and flashing ESP32 binaries

If you have properly set up esp-idf, you should have an alias `get_idf` to a script that sets up environment variables for esp-idf in your shell session. Building the binary then works as follows:

```
get_idf
cd ./esp32_data_collection (or cd ./esp32_inference)
idf.py build
```

After successfully building and connecting everything as described above you can connect the UART adapter to your computer via USB. You might have to first set the right permissions for the device (it is the case for me on Ubuntu):

```
sudo chmod a+rw /dev/ttyUSB0
```

and then you can flash the binary with

```
idf.py -p /dev/ttyUSB0 flash monitor
```

If you do not want to monitor the output, you can omit `monitor`. Conversely, if you just want to establish the serial connection and not flash a new binary, leave out `flash`. Note that sometimes it is necessary to push the reset button on the ESP32 CAM to start flashing. 

## Data Capture

After flashing the data capture binary, connecting everything as described above, and inserting the sdcard, the ESP32 CAM concurrently takes pictures (and optionally IMU data) and writes it to the sdcard. If you have an MPU6050 connected and want to capture the IMU data as well, make sure to set the global `record_imu_data` in `./esp32_data_collection/main/main.cc` to `true` before building.
After each restart (trigger with the reset button or interrupt power supply) a new folder (with incrementing folder name) is created to which the data is written. On a successful initialization, the LED on the ESP32 CAM board should light up for about a second. The resulting folder will for example look like this

```
00001/capture_000000557.gs
00001/capture_000000637.gs
00001/capture_000000717.gs
...
00001/capture_000913557.gs
00001/gyrodata.csv
```

where `capture_*.gs` are 96x96 8-bit grayscale images and `gyrodata.csv` contains the IMU data. The timestamp (time since boot) in the image filename suffix is in the format `hhmmssSSS`. The timestamps in the first column of `gyrodata.csv` are of the same format. With IMU data recording switched on, an IMU record is taken every 20ms and a picture is taken about every 600ms. Without IMU data, an image is recorded about every 300ms.

## Data preprocessing

The following steps are done in python and to run them you first need to install the dependencies (preferably into a virtual env or conda env) by

```
cd ./pothole_model
pip install -r requirements.txt
```

After capturing some data, copy all capture dirs that you want to use from the sdcard into a folder on your computer and point `raw_data_dir` in the config file `./pothole_model/model/config.yaml` to it. 

The original idea for the project was to use the IMU data to generate labels for the images by looking at the strength of the vibrations that occurred in a time window after the image was taken. However, this still requires some work, so I also went with manually labeling the images for now. To start with an easy case, I focused on detecting white markings on the surface ahead, which usually come with bumps in the transition - so it is worth being warned about them. 

In order to manually label the images, they are converted to JPEG. Set `converted_images_dir` in the config to the directory where the converted image data should be written to. Then, run

```
python -m preprocessing.prepare_manual_labeling --config_path=./model/config.yaml
```

which converts all captured images in the configured `raw_data_dir` to JPEG and saves them in `converted_images_dir`. Also, a `labels_template.csv` is written to each capture dir in `raw_data_dir`. It has two columns, one for the filename and one for the label. Edit this file in each capture dir to label the images and rename it to `labels.csv` afterward. Then, to collect, augment and split the data into train, validation and test, set `train_data_dir` in the config and run

```
python -m preprocessing.preprocess_data --config_path=./model/config.yaml
```

which saves the respective data in the `./train`, `./val` and `./test` subdirectories of your './train_data_dir'. There is not any actual preprocessing of the images performed in this step. However, upon loading the data into a `tf.data.Dataset`, the unsigned 8-bit integer values are converted into signed 8-bit integers, because this conversion needs to be done as well during inference, which in turn is necessary since unsigned 8-bit integer quantization is getting [deprecated](https://www.tensorflow.org/lite/performance/quantization_spec#signed_integer_vs_unsigned_integer) in TF Lite. For the data augmentation which is only done for the training data, the images are just vertically flipped. 

## Model Training and Conversion

### Training

The ESP32 CAM platform is quite constrained and therefore an efficient model architecture needs to be used. In this project, we use the original MobileNet architecture:

> A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam\
*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* \
http://arxiv.org/abs/1704.04861

In the config (`./pothole_model/model/config.yaml`) you can set the `model_save_dir`, which is where the trained model will be saved as a SavedModel. You can also set a few other model/training-related parameters, in particular the `alpha` parameter, which controls the number of filters used throughout the model. So by increasing it, you will get a model with more capacity, but also with a larger model size and longer inference time. To train the model run

```
python -m model.train --config_path=./model/config.yaml
```

After training, the script also evaluates the model on the test dataset, which you should only use as an performance estimate for your final model. For the model that is included in this repo, I used ~3800 manually labeled images (before augmentation and splitting) in which the relative frequency of the classes `bump` and `no bump` are 0.4 and 0.6, respectively. The final model achieved an accuracy of `0.97` on the test set.

### Conversion

In order to quantize and convert the SavedModel in `model_save_path`, set the `tflite_model_path` in the config and run:

```
./convert_model_to_tflite.sh
```

This will run the conversion to `pothole.tflite`, which includes the int8 quantization. Quantization can of course impact the performance of the model, therefore it needs to be tested again after the conversion by running:
```
python -m model.evaluate --config_path=./model/config.yaml
```
which evaluates the converted model on the test dataset (change it to validation dataset, if you are still iterating on the model). The model included in this repo also achieves an accuracy of `0.97` after conversion. The next step is to create a C++ file `pothole_model.cc` which defines a byte array that represents the model. This file then needs to be copied into the `./esp32_inference` subproject:

```
cp models/pothole/pothole_model.cc ../esp32_inference/main/pothole_model.cc
```

where it is then included and build into the binary for inference. Note: `./convert_model_to_tflite.sh` still contains some hard-coded paths, which you need to adjust if you want to use different paths/model names. 

## Inference

After building and flashing the inference binary as described above, the model should run at about 2 fps and switch on the buzzer each time a bump/marking is detected until no-bump is detected anymore.

The converted version of the included model (`./esp32_inference/main/pothole_model.cc`), might not work well in your circumstances (different longboard/camera perspective, different surface conditions) so I recommend collecting data and training a new model. If you want to see how it looks like when it works as intended, have a look [here](https://www.youtube.com/watch?v=RgqTGeaQ5AA).

## TODO

- Nicer case, maybe 3d printed 
- Use IMU data for annotation
- Parameterize convert_model_to_tflite script