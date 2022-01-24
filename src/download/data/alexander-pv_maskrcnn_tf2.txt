## Mask-RCNN in Tensorflow v2  ##

This repository is based on [matterport](https://github.com/matterport/Mask_RCNN) Mask-RCNN model implementation. The
main things about the model were added from the original repository. The repo is an attempt to make Mask-RCNN model more
transparent to researchers and more applicable in terms of inference optimization. Besides, new backbones were added in
order to have a choice in balance between accuracy and speed, to make model more task-specific.

### Supported Tensorflow versions

* v2.2.0, v2.3.4, v2.4.3, v2.5.1

### Supported backbones ###

* ResNet [18, 34, 50, 101, 152]
* ResNeXt [50, 101]
* SE-ResNet [18, 34, 50, 101, 152]
* SE-ResNeXt [50, 101]
* SE-Net [154]
* MobileNet [V1, V2]
* EfficientNet [B0, B1, B2, B3, B4, B5, B6, B7]


      Backbone keys:

      'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
      'resnext50', 'resnext101',
      'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
      'seresnext50', 'seresnext101', 'senet154',
      'mobilenet', 'mobilenetv2',
      'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
      'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7',


## Getting started

### Environment setup

```bash
# Define preferred Tensorflow version: tf2.{2,3,4,5}
# For example, Tensorflow 2.2 env:
$ conda create -n tf2.2 python=3.7
$ conda activate tf2.2
$ cd ./requirements && pip install -r requirements_tf2.2.txt

# You may also need onnx_graphsurgeon and tensorrt python binding for TensorRT optimization
$ pip install <TENSORRT_PATH>/python/<TENSORRT_PYTHON_BINDING.whl>
$ pip install <TENSORRT_PATH>/onnx_graphsurgeon/onnx_graphsurgeon-x.y.z-py2.py3-none-any.whl
```

### Prepare dataset class and data augmentation

1. There is a general config about Mask-RCNN building and training in `./src/common/config.py` represented as a dict.
   Prepare it for a specific task (`CLASS_DICT` dictionary for class ids and names, other parameters are in `CONFIG`
   dictionary.)

2. Configure your dataset class. In the basic example we use general dataset class named `SegmentationDataset` for
   dealing with masks made in VGG Image Annotator.\
   In `./src/samples/balloon` you can inspect prepared `BalloonDataset` which inherits `SegmentationDataset` and process
   balloon image samples from the original repository.\
   In `./src/samples/coco` you can inspect prepared `CocoDataset` for MS COCO dataset.\
   Any prepared dataset class can be passed to DataLoader in  `./src/preprocess/preprocess.py` which generates batches.\
   You can also configure your own data augmentation. The default training augmentation is
   in `./src/preprocess/augmentation.py`
   in `get_training_augmentation` function. The default augmentation pipeline is based
   on [albumentations](https://github.com/albumentations-team/albumentations) library.

   See also:
    * `./notebooks/example_data_loader_balloon.ipynb`
    * `./notebooks/example_data_loader_coco.ipynb`

### Training

Basic example:

```python
import tensorflow as tf
from preprocess import preprocess
from preprocess import augmentation as aug
from training import train_model
from model import mask_rcnn_functional
from common.utils import tf_limit_gpu_memory
from common.config import CONFIG

# Limit GPU memory for tensorflow container
tf_limit_gpu_memory(tf, 4500)

# Update info about classes in your dataset
CONFIG.update({'class_dict': {},
               'num_classes':,
},
)
CONFIG.update({'meta_shape': (1 + 3 + 3 + 4 + 1 + CONFIG['num_classes']), })

# Init Mask-RCNN model
model = mask_rcnn_functional(config=CONFIG)

# Init training and validation datasets
base_dir = os.getcwd().replace('src', 'dataset_folder')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_dataset = preprocess.SegmentationDataset(images_dir=train_dir,
                                               classes_dict=CONFIG['class_dict'],
                                               preprocess_transform=preprocess.get_input_preprocess(
                                                   normalize=CONFIG['normalization']
                                               ),
                                               augmentation=aug.get_training_augmentation(),
                                               **CONFIG
                                                )
val_dataset = preprocess.SegmentationDataset(images_dir=val_dir,
                                             classes_dict=CONFIG['class_dict'],
                                             preprocess_transform=preprocess.get_input_preprocess(
                                                 normalize=CONFIG['normalization']
                                             ),
                                             json_annotation_key=None,
                                             **CONFIG
                                             )
# train_model function includes dataset and dataloader initialization, callbacks configuration, 
# a list of losses definition and final model compiling with optimizer defined in CONFIG.
train_model(model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=CONFIG,
            weights_path=None)
```

Balloon dataset example:

Download balloon dataset [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip)

```python
import os

os.chdir('..')
import tensorflow as tf

from samples.balloon import balloon
from preprocess import preprocess
from preprocess import augmentation as aug
from training import train_model
from model import mask_rcnn_functional
from common.utils import tf_limit_gpu_memory

# Limit GPU memory for tensorflow container
tf_limit_gpu_memory(tf, 4500)

from common.config import CONFIG

CONFIG.update(balloon.BALLON_CONFIG)

# Init Mask-RCNN model
model = mask_rcnn_functional(config=CONFIG)

# Init training and validation datasets
base_dir = os.getcwd().replace('src', 'balloon')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_dataset = balloon.BalloonDataset(images_dir=train_dir,
                                       class_key='object',
                                       classes_dict=CONFIG['class_dict'],
                                       preprocess_transform=preprocess.get_input_preprocess(
                                           normalize=CONFIG['normalization']
                                       ),
                                       augmentation=aug.get_training_augmentation(),
                                       json_annotation_key=None,
                                       **CONFIG
                                       )

val_dataset = balloon.BalloonDataset(images_dir=val_dir,
                                     class_key='object',
                                     classes_dict=CONFIG['class_dict'],
                                     preprocess_transform=preprocess.get_input_preprocess(
                                         normalize=CONFIG['normalization']
                                     ),
                                     json_annotation_key=None,
                                     **CONFIG
                                     )

# train_model function includes dataset and dataloader initialization, callbacks configuration, 
# a list of losses definition and final model compiling with optimizer defined in CONFIG.
train_model(model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=CONFIG,
            weights_path=None)
```

See `./notebooks/example_training_balloon.ipynb`.

MS COCO dataset example:

```python
import os

os.chdir('..')
import tensorflow as tf

from samples.coco import coco
from preprocess import preprocess
from preprocess import augmentation as aug
from training import train_model
from model import mask_rcnn_functional
from common.utils import tf_limit_gpu_memory

# Limit GPU memory for tensorflow container
tf_limit_gpu_memory(tf, 4500)

from common.config import CONFIG

CONFIG.update(coco.COCO_CONFIG)

# Init Mask-RCNN model
model = mask_rcnn_functional(config=CONFIG)

# You can also download dataset with auto_download=True argument
# It will be downloaded and unzipped in dataset_dir
base_dir = r'<COCO_PATH>/coco2017'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

train_dataset = coco.CocoDataset(dataset_dir=base_dir,
                                 subset='train',
                                 year=2017,
                                 auto_download=True,
                                 preprocess_transform=preprocess.get_input_preprocess(
                                     normalize=CONFIG['normalization']
                                 ),
                                 augmentation=aug.get_training_augmentation(),
                                 **CONFIG
                                 )

val_dataset = coco.CocoDataset(dataset_dir=base_dir,
                               subset='val',
                               year=2017,
                               auto_download=True,
                               preprocess_transform=preprocess.get_input_preprocess(
                                   normalize=CONFIG['normalization']
                               ),
                               **CONFIG
                               )

train_model(model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=CONFIG,
            weights_path=None)
```

See `./notebooks/example_training_coco.ipynb`.

4. Logs folder with weights and scalars will appear outside `src`. Monitoring training with tensorboard tool:

```bash
$ tensorboard --log_dir=logs
```

### Inference

See inference example in `./notebooks/example_inference_tf_onnx_trt_balloon.ipynb`

### Inference optimization

The project suggests a straightforward way of __Mask-RCNN inference optimization__ on x86_64 architecture and also on
NVIDIA Jetson devices (AArch64). Here you do not need to fix .uff graph and then optimize it with TensorRT. The model
optimizing way here is based on pure .onnx graph with only one prepared .onnx graph modification function for TensorRT.
You can inspect optimization steps with python in `example_tensorflow_to_onnx_tensorrt_balloon.ipynb`.

Optimization steps:

1. Initialize your model in inference mode and load its weights. Thus, your model won't include unnecessary layers that
   is used in training mode.
2. Convert your tensorflow.keras model to .onnx with `tf2onnx`.

__Inference with onnxruntime:__  
From this step, you can use generated .onnx graph in `onnxruntime` and `onnxruntime-gpu` inference.

__Inference with TensorRT:__

3. Change your .onnx graph made on step 2. by including TensorRT-implemented Mask-RCNN layers with
   `onnx-graphsurgeon` library. This step is implemented in `modify_onnx_model` function.
4. Use TensorRT optimization for a modified .onnx-graph to prepare TensoRT-engine:

#### Mask-RCNN with TensorRT >=7.2:

1. Get your TensorRT path: <TENSORRT_PATH>


2. Make sure that the following path in ~/.bashrc:

   `export LD_LIBRARY_PATH=<TENSORRT_PATH>/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

   `export LD_LIBRARY_PATH=<TENSORRT_PATH>/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

   Easy to edit example:

   `export LD_LIBRARY_PATH=/home/user/TensorRT-7.2.3.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

   `export LD_LIBRARY_PATH=/home/user/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`


3. Several layers of MaskRCNN in TensorRT were implemented as special plugins.
One of them is `proposalLayerPlugin` which contains general parameters to be changed. In this repo parameters are placed in
`src/common/config.py`. Thus, to configure MaskRCNN special layers in TensorRT,
it is important to rebuild `nvinfer_plugin` with updated config.

```bash
# Clone TensorRT OSS
$ git clone https://github.com/NVIDIA/TensorRT.git
# Set your TensorRT version by switching the branch. Here is an example for 7.2
$ cd TensorRT/ && git checkout release/7.2 && git pull
$ git submodule update --init --recursive
$ mkdir -p build && cd build
```

3. Open header `TensorRT/plugin/proposalLayerPlugin/mrcnn_config.h` and change Mask-RCNN config according to the trained
   model configuration that is stored in `src/common/config.py`


4. Learn about your compute capabilities: [Your GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).
   For example, Nvidia Geforce RTX 2060 has 7.5, `DGPU_ARCHS=75`.


5. Build `nvinfer_plugin`:
```bash
$ cmake .. -DGPU_ARCHS=75 -DTRT_LIB_DIR=<TENSORRT_PATH>/lib -DTRT_OUT_DIR=`pwd`/out -DCMAKE_C_COMPILER=/usr/bin/gcc
$ make nvinfer_plugin -j$(nproc)
```


6. Copy the `libnvinfer_plugin.so.x.y.z` output to the TensorRT library folder. Don't forget to back up the original build:

```bash
$ mkdir ~/backups
$ sudo mv <TensorRT>/lib/libnvinfer_plugin.so.7.2.3 ~/backups/libnvinfer_plugin.so.7.2.3.bak
$ sudo cp libnvinfer_plugin.so.7.2.3  <TensorRT>/lib/libnvinfer_plugin.so.7.2.3
# Update links
$ sudo ldconfig
# Check that links exist
$ ldconfig -p | grep libnvinfer
```


7.Generate TensorRT-engine in terminal with trtexec:

 * Basically, terminal does not recognize trtexec command. You can add to ~/.bashrc path to trtexec with alias:

   `alias trtexec='<TENSORRT_PATH>/bin/trtexec'`

   For successful run `example_tensorflow_to_onnx_tensorrt_balloon.ipynb` add `TRTEXEC` to ~/.bashrc:

   `export TRTEXEC='<TENSORRT_PATH>/bin/trtexec'`

 * Update ~/.bashrc:

   `$ source ~/.bashrc`

 * Run trtexec:

     * fp32:
       `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --workspace=<WORKSPACE_SIZE> --verbose`
     * fp16:
       `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --fp16 --workspace=<WORKSPACE_SIZE> --verbose`
     
       
#### Mask-RCNN with NVIDIA Jetson devices. TensorRT=7.1.3:

This [NVIDIA doc](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/yolo_v4.html#tensorrt-oss-on-jetson-arm64)
about TensorRT OSS on Jetson was very helpful for the manual:

1. Update cmake on Jetson Ubuntu 18.04 OS:

```bash
$ sudo apt remove --purge --auto-remove cmake
$ wget https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz
$ tar xvf cmake-3.13.5.tar.gz
$ cd cmake-3.13.5/
$ ./configure
$ make -j$(nproc)
$ sudo make install
$ sudo ln -s /usr/local/bin/cmake /usr/bin/cmake
```

2. Clone TensorRT repository to build necessary Mask-RCNN plugins for custom layers:

```bash
$ git clone https://github.com/NVIDIA/TensorRT.git
$ cd TensorRT/ && git checkout release/7.1 && git pull
$ git submodule update --init --recursive
$ export TRT_SOURCE=`pwd`
$ mkdir -p build && cd build
```

3. Open header `TensorRT/plugin/proposalLayerPlugin/mrcnn_config.h` and change Mask-RCNN config according to the trained
   model configuration that is stored in `src/common/config.py`


4. Build `nvinfer_plugin`:

```bash
$ /usr/local/bin/cmake .. -DGPU_ARCHS=72  -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
$ make nvinfer_plugin -j$(nproc)
```

5. Copy the `libnvinfer_plugin.so.7.1.3` output to the library folder. Don't forget to back up the original build:

```bash
$ mkdir ~/backups
$ sudo mv /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3 ~/backups/libnvinfer_plugin.so.7.1.3.bak
$ sudo cp libnvinfer_plugin.so.7.1.3  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.7.1.3
# Update links
$ sudo ldconfig
# Check that links exist
$ ldconfig -p | grep libnvinfer
```

6. Generate TensorRT-engine in terminal with trtexec:
    * You can add to ~/.bashrc path to trtexec with alias if it is not known in terminal:

      `alias trtexec='<TENSORRT_PATH>/bin/trtexec'`

    * Update ~/.bashrc:

      `$ source ~/.bashrc`

    * Run trtexec:

        *
      fp32: `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --workspace=<WORKSPACE_SIZE> --verbose`

        *
      fp16: `trtexec --onnx=<PATH_TO_ONNX_GRAPH> --saveEngine=<PATH_TO_TRT_ENGINE> --fp16 --workspace=<WORKSPACE_SIZE> --verbose`

See inference optimization examples in `./src/notebooks/example_tensorflow_to_onnx_tensorrt_balloon.ipynb`

#### Inference speed comparison with original Mask-RCNN:

Profiling with trtexec TensorRT tool with default maxBatch (1). For tests we took Mask-RCNN model with 2 classes
including background. Note, that for comparison we used original Mask-RCNN model with ResNet101 and (1, 3, 1024, 1024)
input shape and updated tensorflow v2 Mask-RCNN model with all supported backbones and with (1, 1024, 1024, 3), (1, 512,
512, 3) input shapes.

For original matterport Mask-RCNN we went through steps suggested in
[sampleUffMaskRCNN](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleUffMaskRCNN#generating-uff-model)
of official TensorRT github repository. For this model trtexec command:

      $ trtexec --uff=mask_rcnn_resnet101_nchw.uff --uffInput=input_image,3,1024,1024 --output=mrcnn_detection,mrcnn_mask/Sigmoid --workspace=4096 --verbose

For tensorflow v2 Mask-RCNN trtexec command:

      $ trtexec --onnx=maskrcnn_<BACKBONE_NAME>_<WIDTH>_<HEIGHT>_3_trt_mod.onnx  --workspace=4096  --verbose
      $ trtexec --onnx=maskrcnn_<BACKBONE_NAME>_<WIDTH>_<HEIGHT>_3_trt_mod.onnx  --tacticSources=-cublasLt,+cublas --workspace=4096  --verbose

RTX2060:

| Model           |Backbone|Precision|Mean GPU compute, ms|Mean Host latency, ms|Input shape|Total params|
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|
|original Mask-RCNN|ResNet101|fp32|166.032|167.335|(1, 3, 1024, 1024)|64,158,584|
|original Mask-RCNN|ResNet101|fp16|50.594|51.6662|(1, 3, 1024, 1024)|64,158,584|
|Mask-RCNN|ResNet18|fp32|125.903|127.226|(1, 1024, 1024, 3)|32,571,861|
|Mask-RCNN|ResNet18|fp16|46.6753|47.7547|(1, 1024, 1024, 3)|32,571,861|
|Mask-RCNN|ResNet34|fp32|126.272|127.678|(1, 1024, 1024, 3)|42,687,445|
|Mask-RCNN|ResNet34|fp16|49.6903|50.7716|(1, 1024, 1024, 3)|42,687,445|
|Mask-RCNN|ResNet50|fp32|150.751|152.056|(1, 1024, 1024, 3)|45,668,309|
|Mask-RCNN|ResNet50|fp16|54.0631|55.1411|(1, 1024, 1024, 3)|45,668,309|
|Mask-RCNN|ResNet101|fp32|186.64|187.973|(1, 1024, 1024, 3)|64,712,661|
|Mask-RCNN|ResNet101|fp16|58.0508|59.1242|(1, 1024, 1024, 3)|64,712,661|
|Mask-RCNN|MobileNet|fp32|115.363|116.763|(1, 1024, 1024, 3)|24,859,596|
|Mask-RCNN|MobileNet|fp16|40.6769|41.7582|(1, 1024, 1024, 3)|24,859,596|
|Mask-RCNN|MobileNetV2|fp32|114.119|115.486|(1, 1024, 1024, 3)|23,958,348|
|Mask-RCNN|MobileNetV2|fp16|43.8202|44.9006|(1, 1024, 1024, 3)|23,958,348|
|Mask-RCNN|EfficientNetB0|fp32|138.189|139.534|(1, 1024, 1024, 3)|25,786,792|
|Mask-RCNN|EfficientNetB0|fp16|56.5004|57.5949|(1, 1024, 1024, 3)|25,786,792|
|Mask-RCNN|EfficientNetB1|fp32|134.059|135.417|(1, 1024, 1024, 3)|28,312,460|
|Mask-RCNN|EfficientNetB1|fp16|60.3303|61.4217|(1, 1024, 1024, 3)|28,312,460|
|Mask-RCNN|EfficientNetB2|fp32|135.788|137.12|(1, 1024, 1024, 3)|29,563,134|
|Mask-RCNN|EfficientNetB2|fp16|64.0362|65.1281|(1, 1024, 1024, 3)|29,563,134|
|Mask-RCNN|EfficientNetB3|fp32| | |(1, 1024, 1024, 3)|32,647,732|
|Mask-RCNN|EfficientNetB3|fp16| | |(1, 1024, 1024, 3)|32,647,732|
|Mask-RCNN|ResNet18|fp32|53.5696|53.9976|(1, 512, 512, 3)|31,786,197|
|Mask-RCNN|ResNet18|fp16|19.6023 |19.941|(1, 512, 512, 3)|31,786,197|
|Mask-RCNN|ResNet34|fp32|59.9331|60.4002|(1, 512, 512, 3)|41,901,781|
|Mask-RCNN|ResNet34|fp16|23.7166|24.063|(1, 512, 512, 3)|41,901,781|
|Mask-RCNN|ResNet50|fp32|65.8216|66.2745|(1, 512, 512, 3)|44,882,645|
|Mask-RCNN|ResNet50|fp16|25.6267|26.0099|(1, 512, 512, 3)|44,882,645|
|Mask-RCNN|ResNet101|fp32|77.0433|77.48|(1, 512, 512, 3)|63,926,997|
|Mask-RCNN|ResNet101|fp16|28.1458|28.498|(1, 512, 512, 3)|63,926,997|
|Mask-RCNN|MobileNet|fp32|52.2146|52.6336|(1, 512, 512, 3)|24,073,932|
|Mask-RCNN|MobileNet|fp16|19.5832|19.9254|(1, 512, 512, 3)|24,073,932|
|Mask-RCNN|MobileNetV2|fp32|52.5706|53.0006|(1, 512, 512, 3)|23,172,684|
|Mask-RCNN|MobileNetV2|fp16|21.9402|22.2757|(1, 512, 512, 3)|23,172,684|
|Mask-RCNN|EfficientNetB0|fp32|57.0875|57.5132|(1, 512, 512, 3)|25,001,128|
|Mask-RCNN|EfficientNetB0|fp16|24.5434|24.8687|(1, 512, 512, 3)|25,001,128|
|Mask-RCNN|EfficientNetB1|fp32|59.3512|59.7616|(1, 512, 512, 3)|27,526,796|
|Mask-RCNN|EfficientNetB1|fp16|22.6646|23.0058|(1, 512, 512, 3)|27,526,796|
|Mask-RCNN|EfficientNetB2|fp32|67.8534|68.2614|(1, 512, 512, 3)|28,777,470|
|Mask-RCNN|EfficientNetB2|fp16|31.5452|31.8778|(1, 512, 512, 3)|28,777,470|
|Mask-RCNN|EfficientNetB3|fp32|68.9046|69.3455|(1, 512, 512, 3)|31,862,068|
|Mask-RCNN|EfficientNetB3|fp16|34.7724|35.0879|(1, 512, 512, 3)|31,862,068|

Jetson AGX Xavier:

|Model|Backbone|Precision|Mean GPU compute, ms|Mean Host latency, ms|Input shape|Total params|
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|
|original Mask-RCNN|ResNet101|fp32|429.839|430.213|(1, 3, 1024, 1024)|64,158,584| |
|original Mask-RCNN|ResNet101|fp16|132.519|132.902|(1, 3, 1024, 1024)|64,158,584| |
|Mask-RCNN|ResNet18|fp32|301.87|302.241|(1, 1024, 1024, 3)|32,571,861|
|Mask-RCNN|ResNet18|fp16|120.743|121.131|(1, 1024, 1024, 3)|32,571,861|
|Mask-RCNN|ResNet34|fp32|326.506|326.893|(1, 1024, 1024, 3)|42,687,445|
|Mask-RCNN|ResNet34|fp16|122.724|123.11|(1, 1024, 1024, 3)|42,687,445|
|Mask-RCNN|ResNet50|fp32|375.936|376.317|(1, 1024, 1024, 3)|45,668,309|
|Mask-RCNN|ResNet50|fp16|130.978|131.368|(1, 1024, 1024, 3)|45,668,309|
|Mask-RCNN|ResNet101|fp32|470.027|470.423|(1, 1024, 1024, 3)|64,712,661|
|Mask-RCNN|ResNet101|fp16|158.226|158.623|(1, 1024, 1024, 3)|64,712,661|
|Mask-RCNN|MobileNet|fp32|291.818|292.217|(1, 1024, 1024, 3)|24,859,596|
|Mask-RCNN|MobileNet|fp16|108.538|108.926|(1, 1024, 1024, 3)|24,859,596|
|Mask-RCNN|MobileNetV2|fp32|285.315|285.688|(1, 1024, 1024, 3)|23,958,348|
|Mask-RCNN|MobileNetV2|fp16|115.311|115.706|(1, 1024, 1024, 3)|23,958,348|
|Mask-RCNN|EfficientNetB0|fp32|320.68|321.056|(1, 1024, 1024, 3)|25,786,792|
|Mask-RCNN|EfficientNetB0|fp16|145.32|145.709|(1, 1024, 1024, 3)|25,786,792|
|Mask-RCNN|EfficientNetB1|fp32|339.343|339.724|(1, 1024, 1024, 3)|28,312,460|
|Mask-RCNN|EfficientNetB1|fp16|154.464|154.837|(1, 1024, 1024, 3)|28,312,460|
|Mask-RCNN|EfficientNetB2|fp32|344.166|344.554|(1, 1024, 1024, 3)|29,563,134|
|Mask-RCNN|EfficientNetB2|fp16|156.596|156.982|(1, 1024, 1024, 3)|29,563,134|
|Mask-RCNN|EfficientNetB3|fp32| | |(1, 1024, 1024, 3)|32,647,732|
|Mask-RCNN|EfficientNetB3|fp16| | |(1, 1024, 1024, 3)|32,647,732|
|Mask-RCNN|ResNet18|fp32|147.313|147.43|(1, 512, 512, 3)|31,786,197|
|Mask-RCNN|ResNet18|fp16|55.0673|55.1861|(1, 512, 512, 3)|31,786,197|
|Mask-RCNN|ResNet34|fp32|160.904|161.024|(1, 512, 512, 3)|41,901,781|
|Mask-RCNN|ResNet34|fp16|62.6873|62.8085|(1, 512, 512, 3)|41,901,781|
|Mask-RCNN|ResNet50|fp32|176.807|176.925|(1, 512, 512, 3)|44,882,645|
|Mask-RCNN|ResNet50|fp16|68.0678|68.1877|(1, 512, 512, 3)|44,882,645|
|Mask-RCNN|ResNet101|fp32|200.177|200.301|(1, 512, 512, 3)|63,926,997|
|Mask-RCNN|ResNet101|fp16|73.7332|73.8529|(1, 512, 512, 3)|63,926,997|
|Mask-RCNN|MobileNet|fp32|143.371|143.492|(1, 512, 512, 3)|24,073,932|
|Mask-RCNN|MobileNet|fp16|52.5975|52.7168|(1, 512, 512, 3)|24,073,932|
|Mask-RCNN|MobileNetV2|fp32|143.504|143.623|(1, 512, 512, 3)|23,172,684|
|Mask-RCNN|MobileNetV2|fp16|54.7317|54.85|(1, 512, 512, 3)|23,172,684|
|Mask-RCNN|EfficientNetB0|fp32|157.063|157.185|(1, 512, 512, 3)|25,001,128|
|Mask-RCNN|EfficientNetB0|fp16|66.0013|66.1224|(1, 512, 512, 3)|25,001,128|
|Mask-RCNN|EfficientNetB1|fp32|158.944|159.064|(1, 512, 512, 3)|27,526,796|
|Mask-RCNN|EfficientNetB1|fp16|65.623|65.7444|(1, 512, 512, 3)|27,526,796|
|Mask-RCNN|EfficientNetB2|fp32|175.904|176.023|(1, 512, 512, 3)|28,777,470|
|Mask-RCNN|EfficientNetB2|fp16|82.7281|82.8464| (1, 512, 512, 3)|28,777,470|
|Mask-RCNN|EfficientNetB3|fp32|184.948|185.083|(1, 512, 512, 3)|31,862,068|
|Mask-RCNN|EfficientNetB3|fp16|83.1854|83.3059|(1, 512, 512, 3)|31,862,068|

### TODOs:

---
* [ ] TRT-models profiling;
* [ ] NCWH support;
* [ ] Mixed precision training;
* [ ] Pruning options;
* [ ] Flexible backbones configuration;
* [ ] Update inference speed test tables;
* [ ] MS COCO weights;
* [ ] Tensorflow v2.6 support;

---

### Changelog

[Link to Changelog](CHANGELOG.md)

### Contributors

Alexander Popkov: [@alexander-pv](https://github.com/alexander-pv)

Feel free to write me about the repo issues and its update ideas.

