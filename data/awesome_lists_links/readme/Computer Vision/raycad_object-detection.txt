## Object Detection Using Deep-Learning Faster-RCNN

- The goal of this tutorial is to guide you how to apply Faster-RCNN to build and train step by step a model for the specific problem.

- Regarding the concept of Faster-RCNN please take a look at the paper links: 

    https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
    
    https://arxiv.org/pdf/1506.01497.pdf

- The main source code is located in the folder: **object-detection/research/object_detection**

#### PROBLEM

In **QC Process** we need to manually inspect the products to verify if the given icons on the products shown or not after performing an action. This tutorial will guide you how to apply **Faster RCNN** to automate testing HMI by recognizing the given icons shown or not in each test case.

### 1. Download and Install Anaconda
```
# Download the latest Anaconda
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

$ chmod a+x Anaconda3-2019.03-Linux-x86_64.sh

# Install the Anaconda 
$ ./Anaconda3-2019.03-Linux-x86_64.sh

# Activating the Installation
$ source ~/.bashrc

# Install essential development packages
$ sudo apt install build-essential
```

### 2. Clone Google Tensorflow Models
```
$ git clone https://github.com/tensorflow/models.git tensorflow_models
```

### 3. Clone Object Detection Training Source Code
```
$ git clone https://github.com/raycad/object-detection.git
```

### 4. Update the Latest Object Detection Code 
Copy all source code from "object_detection_tutorial" to the "object-detection/research/object_detection" directory
```
$ cp -rf object_detection_tutorial/* object-detection/research/object_detection
```

### 5. Create a New Virtual Environment
```
$ conda create -n tensorflow_cpu pip python=3.6

# Activate the newly created virtual environment
$ conda activate tensorflow_cpu

# Deactivate created virtual environment
$ conda deactivate
```

### 6. Install Dependencies 
```
# [NOTE]: Install one by one package
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ conda install -c anaconda protobuf
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install pillow
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install lxml
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install Cython
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install jupyter
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install matplotlib
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install pandas
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install opencv-python
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install pycocotools

# Upgrade the tensorflow. It works with tensorflow 1.13.0
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ pip install --upgrade tensorflow

$ pip list | grep tensorflow	
	tensorflow           1.13.1  
	tensorflow-estimator 1.13.0  
```

### 7. Configure PYTHONPATH Environment Variable

**NOTE:** Every time the **"tensorflow_cpu"** virtual environment is exited, the **PYTHONPATH** variable is reset and needs to be set up again

```
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research$ export PYTHONPATH=~/dev/tensorflow/object-detection:~/dev/tensorflow/object-detection/research:~/dev/tensorflow/object-detection/research/slim
```

### 8. Compile and Run Protobufs
```
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research$ protoc ./object_detection/protos/*.proto --python_out=.
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research$ python setup.py build
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research$ python setup.py install

# Test TensorFlow setup to verify it works
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ jupyter notebook object_detection_tutorial.ipynb
```

### 9. Gather and Label Pictures
Use LabelImg (https://github.com/tzutalin/labelImg) to label and make annotation images

```
# You can check if the size of each bounding box is correct by running size_checker.py
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python size_checker.py --move
```

![Label Images](./doc/label_images.png)

### 10. Generate Training Data
This creates a train_labels.csv and test_labels.csv file in the /object_detection/images folder
```
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python xml_to_csv.py
```

### 11. Generate Tensorflow Records
Open the generate_tfrecord.py file in a text editor then replace the label map starting at line 31 with your own label map, where each object is assigned an ID number
```
# This same number assignment will be used when configuring the labelmap.pbtxt file
# Then, generate the TFRecord files by issuing these commands from the /object_detection folder
# These generate a train.record and a test.record file in "/object_detection". These will be used to train the new object detection classifier.
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```

### 12. Use the Faster-RCNN-Inception-V2 Model
Download the model from http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
# Extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the "object_detection" folder
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ tar -zxvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

# Create Label Map and Configure Training
# Update information of the files in the "object-detection/research/object_detection/training" folder. Note to set the absolute path for "fine_tune_checkpoint", "train_input_reader" and "eval_input_reader"

Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .

fine_tune_checkpoint: "/home/seedotech/dev/tensorflow/object-detection/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

train_input_reader: {
  tf_record_input_reader {
    input_path: "/home/seedotech/dev/tensorflow/object-detection/research/object_detection/train.record"
  }
  label_map_path: "/home/seedotech/dev/tensorflow/object-detection/research/object_detection/training/labelmap.pbtxt"
}

eval_config: {
  # Change num_examples to the number of images you have in the /images/test directory.
  num_examples: 16
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "/home/seedotech/dev/tensorflow/object-detection/research/object_detection/test.record"
  }
  label_map_path: "/home/seedotech/dev/tensorflow/object-detection/research/object_detection/training/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
}
```

### 13. Train the Object Detection Model
```
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

# Run with old train.py"
# Move train.py from /object_detection/legacy into the /object_detection folder and then continue following the steps below
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ cp legacy/train.py .
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

![Train Model](./doc/train_model.png)

### 14. Export Inference Graph
```
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the /object_detection folder, issue the following command, where “XXXX” in "model.ckpt-XXXX" should be replaced with the highest-numbered .ckpt file in the training folder:

(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

e.g.
(tensorflow_cpu) seedotech@tensorflow:~/dev/tensorflow/object-detection/research/object_detection$ python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-10000 --output_directory inference_graph

This creates a frozen_inference_graph.pb file in the /object_detection/inference_graph folder. The .pb file contains the object detection classifier.
```

### 15. Run Object Detection Example
Edit the **line 51** in the **object_detection_image.py** file to the number of classes the object detector can identify

```
$ python object_detection_image.py
```

![Result](./doc/ret1.png)

### 16. Common Issues
#### 16.1. File "/home/seedotech/anaconda3/envs/tensorflow_cpu/lib/python3.6/site-packages/object_detection-0.1-py3.6.egg/object_detection/utils/learning_s
chedules.py", line 160, in manual_stepping
    raise ValueError('First step cannot be zero.')
ValueError: First step cannot be zero.

**[FIX]** 
```
$ nano utils/learning_schedules.py

Then uncomment 2 lines in line 160:

#if boundaries and boundaries[0] == 0:
#  raise ValueError('First step cannot be zero.')
```

#### 16.2. Training process was killed without throwing any problems
```
INFO:tensorflow:global step 5778: loss = 0.0613 (2.646 sec/step)
INFO:tensorflow:global step 5778: loss = 0.0613 (2.646 sec/step)
INFO:tensorflow:global step 5779: loss = 0.0551 (2.673 sec/step)
INFO:tensorflow:global step 5779: loss = 0.0551 (2.673 sec/step)
INFO:tensorflow:global step 5780: loss = 0.0558 (2.737 sec/step)
INFO:tensorflow:global step 5780: loss = 0.0558 (2.737 sec/step)
INFO:tensorflow:global step 5781: loss = 0.0759 (2.663 sec/step)
INFO:tensorflow:global step 5781: loss = 0.0759 (2.663 sec/step)
INFO:tensorflow:global step 5782: loss = 0.0798 (2.706 sec/step)
INFO:tensorflow:global step 5782: loss = 0.0798 (2.706 sec/step)
INFO:tensorflow:global step 5783: loss = 0.0396 (2.661 sec/step)
INFO:tensorflow:global step 5783: loss = 0.0396 (2.661 sec/step)
Killed
```

It was caused by out-of-memory. Please check your RAM consumption.

### 17. References

https://github.com/tensorflow/models

https://cloud.google.com/solutions/creating-object-detection-application-tensorflow

https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example/blob/master/object_detection_app/app.py

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html