## BrandLOGO_detection
Brand logo detection on custom dataset using Tesorflow object detection API.

## Folder Structure
- BrandLOGO_detection
  - pre_trained_models
    - *downloaded files for the choosen pre-trained model will come here* 
  - Dataset
    - Annotations
      - *Annotations for your training images will come here*
    - train_images
      - *all of your images for training will come here*
    - test_data
      - *all your images for testing will come here*
    - lable_map.pbtxt
    - train.tfrecord
    - test.tfrecord
   - Inf_Graph
     - *inference graph of the trained model will be saved here*
   - Checkpoint
     - *checkpoints of the trained model will be saved here*
   - BrandLogo_detection.ipynb
   - *config file for the choosen model*

## Description
Tensorflow object detection is very good, easy and free framework for object detection tasks.

It contains varous models trained on COCO, Kitty like large dataset and we use these pretrained models on our custom datset.


## Detection Results

These are some detection results by DeepLogo.

|||
|---|---|
|![example1](model_results/res1.png)|![example2](model_results/res2.jpg)|
|![example3](model_results/res3.png)|![example4](model_results/sample_res4.png)|

## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [steps](#installation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## About the Project

### problem statement
Brand logo detection system custom dataset using Tesorflow object detection API.

### Architecture
![example5](Architecture/index.jpg)

## Getting Started

### Prerequisites and Steps

1. Dependancies 
    ```sh
    pip install python==3.6
    pip install virtualenv 
    pip install -r requirements.txt
    ```
you may need to download and install other packages also as you go.
** Do not install tensorflow 2.0. **

2. Clone the tensorflow/models repository and download the pre-trained model from model zoo.
     ```sh
     git clone https://github.com/tensorflow/models.git
     cd models/research
     python setup.py install
     export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
     protoc object_detection/protos/*.proto --python_out=.
     ```
windows users need to build protobuf from source and also need to download VScode with c++ support.

3. clone this repo or create structure like this in your local enviournment.

4. Download dataset or create custom dataset
    Download the flickr logos 27 dataset from [here](http://image.ntua.gr/iva/datasets/flickr_logos/) and unzip it and move to
    Dataset folder
    Or
    For custom dataset capture images, Once you have captured images, transfer it to your PC and resize it to a smaller size 
    so that your training will go smoothly without running out of memory. Now rename and divide your captured images into two 
    chunks, one chunk for training(80%) and other for testing(20%). Finally, move training images into train_data folder and
    testing images into test_data folder
  
5. Label the data
    for labelling your custom data you need to download [LableImg](https://github.com/tzutalin/labelImg) tool.
    Label the data and save it in Annotations folder.
  
6. Preprocess the annotation files
    flickr training Annotations file contains some invalid annotations.
   ```sh
     cd BrandLOGO_detection 
     python preprocessor.py
     ```
   
7. Generate tfrecord files.

     The Tensorflow Object Detection API expects data to be in the TFRecord format. Run the following command to convert from       
     preprocessed files into TFRecords.

     In case you made custom dataset:
     ```sh
     python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=<path_to_your_dataset_directory> --annotations_dir=<name_of_annotations_directory> --output_path=<path_where_you_want_record_file_to_be_saved> --label_map_path=<path_of_label_map_file>
     ```

     for fickr dataset:
     ```
      python tfrecord_generator.py --train_or_test train --csv_input Dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation_cropped.txt --img_dir Dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path Dataset/train.tfrecord

      python gen_tfrecord.py --train_or_test test --csv_input Dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_test_set_annotation_cropped.txt --img_dir Dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images --output_path Dataset/test.tfrecord ```
    

8. **Training** 
    1. Download the pretrained model you want from [Tensorflow detection model zoo Github page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

    2.Extract it.

    3.keep it in pretrained model folder. now,

    ```sh 
    cd models/research/object_detection/sample/config/
    ```

    5.copy model to main directory(BrandLOGO_detection).

    6.open that .config file and search "PATH TO BE CONFIGURED" and change it with required path. now,

    ```sh 
    cd models/research/ 

    python object_detection/legacy/train.py --train_dir=<path_to_the folder_for_saving_checkpoints> --pipeline_config_path=<path_to_config_file> 
    ```

    Example:
    
    ```sh python object_detection/legacy/train.py --train_dir=<full dir>/BrandLOGO_detection/checkpoint --pipeline_config_path=/BrandLOGO_detection/faster_rcnn_resnet101_coco.config
    ```
  interrurpt training when loss is below 0.1.Checkpoints will be saved in Checkpoint folder

9. generate inference graph from saved checkpoints

    ```sh
    python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=<path_to_config_file> --trained_checkpoint_prefix=<path to saved checkpoint> --output_directory=<path_to_the_folder_for_saving_inference_graph>
    ```
it will be saved in inf_graph folder.

10. **Testing**

    ```sh 
    cd models/research/object_detection/object_detection_tutorial.ipynb
    ```

    2. make nessesary changes which results in BrandLOGOdetector.ipynb as given.

    3. you just need to change paths and class number in BrandLOGOdetector.ipynb and run it!

    4. you can see evalation result on ** Tensorboard **
  
## Acknowledgements
    https://towardsdatascience.com/training-a-tensorflow-faster-r-cnn-object-detection-model-on-your-own-dataset-b3b175708d6d

    https://www.analyticsvidhya.com/blog/2020/04/build-your-own-object-detection-model-using-tensorflow-api/

    https://github.com/tensorflow/models/tree/master/research/object_detection

    https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b
  
    Y. Kalantidis, LG. Pueyo, M. Trevisiol, R. van Zwol, Y. Avrithis. Scalable Triangulation-based Logo Recognition. In Proceedings of ACM International Conference on Multimedia Retrieval (ICMR 2011), Trento, Italy, April 2011.
    https://arxiv.org/abs/1512.02325
  
## License
MIT
