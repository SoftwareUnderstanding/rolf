# Waymo Open Dataset: Tensorflow 2 Object Detection Development Record
contributed by < `gyes00205` >
###### tags: `waymo`

## Download Dataset
[Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0)
**domain_adaptation** directory doesn't have label data, so please download the data in **training** directory

## tfrecord details
* 5 kinds of camera photos and Lidar informations
* num_classes: 0: Unknown, 1: Vehicle, 2: Pedestrian, 3: Sign, 4: Cyclist 
    In this project, we don't need sign and Unknown classes, so we should modify label_map.pbtxt :
```pbtxt 
item {
    id: 1
    name: 'vehicle'
}

item {
    id: 2
    name: 'pedestrian'
}

item {
    id: 4
    name: 'cyclist'
}
```
* camera categories: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
    <img src="https://i.imgur.com/Q68Lepf.jpg">
* bbox (x, y, w, h) coordinate: (x, y) represents center coordinate of bbox, (w, h) represents width and height.

## Setup Environment
### Install Waymo open dataset
```shell 
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
```
### Install COCO API
```shell 
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
### Install Tensorflow 2 Object Detection API
Refer to [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) to install toolkits
* git clone Tensorflow 2 Object Detection API
```shell 
git clone https://github.com/tensorflow/models.git
```
* go to models/research/ and run
```shell 
protoc object_detection/protos/*.proto --python_out=.
```
* add API to your environment path
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
* copy setup.py to models/research/
```shell 
cp object_detection/packages/tf2/setup.py ./
```
* install setup.py
```shell 
python -m pip install .
```
* Test whether the installation is successful
```
python object_detection/builders/model_builder_tf2_test.py
```
### Build directory structure
```
Waymo
├───models #Tensorflow Object Detection API
├───training_configs #training config
├───pre-trained-models #pretrained model
├───exported-models #exported model
└───data #training data
    └───segment-???.tfrecord
```
##  Convert tfrecord format
Besides of Lidar informations in waymo's tfrecord, the below is its bbox format:

(x0, y0): is center coordinate. (w, h): is width and height.

![](https://i.imgur.com/WSDKAQZ.png)

Our goal is to filter out Lidar and convert bbox to the following format:

(x1, y1): is left-top coordinate. (x2, y2): is right-down coordinate.

![](https://i.imgur.com/HyR6xS0.png)

The reference code that convert tfrecord is [LevinJ/tf_obj_detection_api](https://github.com/LevinJ/tf_obj_detection_api), and make some minor changes.

**create_record.py:**

filepath: the path of tfrecord 

data_dir: the converted tfrecord will be stored in the data_dir/processed directory

Execute code:
    
```shell 
python create_record.py \
--filepath=data/segment-???.tfrecord \
--data_dir=data/
```
    
After executing th code, the processed tfrecord will appear in data/processed directory.
```
Waymo
├───models
├───training_configs 
├───pre-trained-models 
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord # processed tfrecord
    └───segment-???.tfrecord
```

## Download pretrained model
go to [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and download pretrained model.

![](https://i.imgur.com/x34zpZL.png)

I download `SSD ResNet50 V1 FPN 640x640 (RetinaNet50)` pretrained model. 
* go to pre-trained-models directory.

`cd pre-trained-models`

* download SSD ResNet50 pretrained model

```
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

* unzip the file

`tar zxvf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz`

```
Waymo
├───models
├───training_configs 
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord #processed tfrecord
    └───segment-???.tfrecord
```

## Modify training config
Go to [configs/tf2](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2), and find corresponding config that is ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config

* Create folder in training_configs directory

```shell 
cd training_configs
mkdir ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
```

* Create pipeline.config in ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 directory. Copy and paste the config content you just found, and make some modifications.
    * num_classes: number of classes 
    * batch_size: according to your computer memory
    * fine_tune_checkpoint: modify to pretrained model ckpt-0 path
    * num_steps: training steps
    * use_bfloat16: whether to use tpu, if not used, set to false
    * label_map_path: label_map.pbtxt path
    * train_input_reader: set input_path to the tfrecord path for training
    * metrics_set: "coco_detection_metrics"
    * use_moving_averages: false
    * eval_input_reader: Set input_path to the tfrecord path for evaluating

```config
# SSD with Resnet 50 v1 FPN feature extractor, shared box predictor and focal
# loss (a.k.a Retinanet).
# See Lin et al, https://arxiv.org/abs/1708.02002
# Trained on COCO, initialized from Imagenet classification checkpoint
# Train on TPU-8
#
# Achieves 34.3 mAP on COCO17 Val

model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: 3 # 3 kinds of classes
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    encode_background_as_zeros: true
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: [1.0, 2.0, 0.5]
        scales_per_octave: 2
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        depth: 256
        class_prediction_bias_init: -4.6
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.0004
            }
          }
          initializer {
            random_normal_initializer {
              stddev: 0.01
              mean: 0.0
            }
          }
          batch_norm {
            scale: true,
            decay: 0.997,
            epsilon: 0.001,
          }
        }
        num_layers_before_predictor: 4
        kernel_size: 3
      }
    }
    feature_extractor {
      type: 'ssd_resnet50_v1_fpn_keras'
      fpn {
        min_level: 3
        max_level: 7
      }
      min_depth: 16
      depth_multiplier: 1.0
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.0004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          scale: true,
          decay: 0.997,
          epsilon: 0.001,
        }
      }
      override_base_feature_extractor_hyperparams: true
    }
    loss {
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.25
          gamma: 2.0
        }
      }
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    normalize_loc_loss_by_codesize: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  fine_tune_checkpoint_version: V2
  #pretrained model ckpt-0 path
  fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" 
  fine_tune_checkpoint_type: "detection" # set to detection
  batch_size: 2
  sync_replicas: true
  startup_delay_steps: 0
  replicas_to_aggregate: 8
  use_bfloat16: false # if not use tpu, set to false
  num_steps: 6000 # training steps
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: .04
          total_steps: 25000
          warmup_learning_rate: .013333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
}

train_input_reader: {
  label_map_path: "./label_map.pbtxt"
  tf_record_input_reader {
    input_path: "data/processed/*.tfrecord"
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}

eval_input_reader: {
  label_map_path: "./label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "data/processed/*.tfrecord"
  }
}
```
```
Waymo
├───models
├───training_configs 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       └───pipeline.config # create pipeline.config
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
└───data
    ├───processed
    │   └───segment-???.tfrecord
    └───segment-???.tfrecord
```
## Train Model

**model_main_tf2.py**

model_dir: the training checkpoint will be stored in the model_dir directory

pipeline_config_path: pipeline.config path

Execute code: 

```shell 
python model_main_tf2.py \
--model_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 \
--pipeline_config_path=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config
```
Execution results: it will be printed every 100 steps.
```
Step 2100 per-step time 0.320s
INFO:tensorflow:{'Loss/classification_loss': 0.121629156,
 'Loss/localization_loss': 0.16370133,
 'Loss/regularization_loss': 0.2080817,
 'Loss/total_loss': 0.4934122,
 'learning_rate': 0.039998136}
I0605 08:29:04.605577 139701982308224 model_lib_v2.py:700] {'Loss/classification_loss': 0.121629156,
 'Loss/localization_loss': 0.16370133,
 'Loss/regularization_loss': 0.2080817,
 'Loss/total_loss': 0.4934122,
 'learning_rate': 0.039998136}
```

## Evaluate Model (Optional)
**model_main_tf2.py**

checkpoint_dir: the directory to read checkpoint.

Execute code: 

```shell 
python model_main_tf2.py \
--model_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 \
--pipeline_config_path=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config \
--checkpoint_dir=training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/ 
```
Execution results: calculate AP and AR

$AP^{small}:$ AP for small object : area < $32^2$

$AP^{medium}:$ AP for medium object : $32^2$ < area < $96^2$ 

$AP^{large}:$ AP for large object : $96^2$ < area

![](https://i.imgur.com/RjN2dRf.png)

## Export Model
**exporter_main_v2.py**

input_type: image_tensor

pipeline_config_path:  pipeline.config path

trained_checkpoint_dir: the path to store checkpoint

output_directory: exported model path

Execute code: 

```shell 
!python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config \
--trained_checkpoint_dir training_configs/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/  \
--output_directory exported-models/my_model_6000steps
```
Execution results:
```shell 
INFO:tensorflow:Assets written to: exported-models/my_model_6000steps/saved_model/assets
I0605 09:07:21.034602 139745385867136 builder_impl.py:775] Assets written to: exported-models/my_model_6000steps/saved_model/assets
INFO:tensorflow:Writing pipeline config file to exported-models/my_model_6000steps/pipeline.config
I0605 09:07:22.310333 139745385867136 config_util.py:254] Writing pipeline config file to exported-models/my_model_6000steps/pipeline.config
```
```
Waymo
├───models
├───training_configs 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       └─pipeline.config # create pipeline.config
├───pre-trained-models 
│   └───ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
│       ├─ checkpoint/
│       ├─ saved_model/
│       └─ pipeline.config
├───exported-models 
│   └───my_model_6000steps
└───data
    ├───processed
    │   └─segment-???.tfrecord
    └───segment-???.tfrecord
```
## Use model to predict
**detect.py**

saved_model_path: exported model path

test_path: image path

output_path: output predicted image path

min_score_thresh: confidience

Execute code:

```shell 
!python detect.py \
--saved_model_path=exported-models/my_model_6000steps \
--test_path=test_image \
--output_path=output_image \
--min_score_thresh=.1
```
Execution results:

<img src="https://i.imgur.com/NNE6OuI.png" width=250px height=200px> 
<img src="https://i.imgur.com/dyRuUpA.png" width=300px height=200px>
<img src="https://i.imgur.com/vICSrnI.png" width=250px height=200px>
<img src="https://i.imgur.com/it53kPf.png" width=300px height=200px>

## Reference
1. [LevinJ/tf_obj_detection_api](https://github.com/LevinJ/tf_obj_detection_api)
2. [Waymo Open Dataset](https://waymo.com/open/)
3. [Waymo quick start tutorial](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb)
4. [Tensorflow Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)