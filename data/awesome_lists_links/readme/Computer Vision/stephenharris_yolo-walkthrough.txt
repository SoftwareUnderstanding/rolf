# Automatic Meter Reading

This repository details how to set up a convolutional neural network to automatically extract meter readings from photos of meters.

## Local set-up

On the local machine run

    xhost +local:docker

    docker-compose up --build

Then, to test, on the running container

    cd darknet
    ./darknet

You should get the output:

    usage: ./darknet <function>

To test the OpenCV/Display run

    ./darknet imtest data/eagle.jpg

(you should see eagles - but this will fail if you're running it on a headless EC2 instance.)

## Running some tests
 
Download some weights:

    mkdir weights
    wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

Then, in `/home/jovyan/darknet`

    ./darknet detect cfg/yolov3.cfg ../weights/yolov3.weights data/dog.jpg

## Project walkthrough

### Preparing your data

For this you will need source images of meters. Once collected, I ran the following to give standardised names to the image files:

    docker exec -it yolo_yolo_1 bash

    # Then in the container
    python util/rename-files.py /path/to/images/of/meters/

    # You can then also check for duplicate images
    python util/find-duplicates.py

I then renamed the images of meters with digital displays by appending a `d` to the filename to ensure that my training, testing and validation set had similar proportion of digital to non-digital displays.

#### Labelling counter  

For each image there needs to be `.txt` file, with the same filename, which labels the data (in particular, the width, height and co-ordinates of the centre of each bounding box (all given as ratios of the image's height/width), and the class number of the object contained by the box). Each line in this file corresponds to an object in the corresponding image (for this step, it should just be the one line).

    <class number box1> <box1 cx ratio> <box1 cy_ratio> <box1 width ratio> <box1 height ratio>
    <class number box2> <box2 cx ratio> <box2 cy ratio> <box2 width ratio> <box2 height ratio>

To achieve this I used https://github.com/tzutalin/labelImg

    xhost +local:docker
    docker run --rm -it -e DISPLAY=unix$DISPLAY --workdir=/home -v "${PWD}:/home" -v /tmp/.X11-unix:/tmp/.X11-unix tzutalin/py2qt4

    # Then inside the container
    python labelImg.py


#### Splitting the trainnig, testing and validation data

This step involves splitting your data set into three groups:

 - **training** - The portion dataset that is used to train you model
 - **testing** - A portion of the dataset used to evaluate different models and their parameters
 - **validation** - The portion of the dataset reserved to give an indication of performance of your final model.

The following will create a `training`, `testing` and `validation` folders and move each image and its annotations into one of them according to the proportions set in the code (I've gone for a 60-20-20 split)

    docker exec -it yolo_yolo_1 bash

    # Then in the container
    python util/split-train-test-data.py /path/to/images/of/meters/


**Note:** When running training, the Darknet yolo implementation, when looking for an image's annotation file, will take the image path, and replace  "images", and the `.jpg` extension with a `.txt` extension. For this reason, avoid using `images` in the path if your annotations live in the same folder as the corresponding image)


### Training counter recognition (on AWS EC2)

Most of this applies when you're running it locally or on some other cloud service, but details may vary slightly).

#### Pre-requisite set up

See `terraform`. This creates:

 - EC2 instance (`p2.xlarge`, with CUDA installed)
 - 2 S3 buckets to store training data, initial weights and configs

Additionally for deploying our trained model

 - 1 S3 bucket to store images passed to the API, along with your model's predictions
 - Lambda which runs our inference
 - API Gateway which provides an API interface for the lambda

(and all the necessary security groups / policies etc)

The `setup.sh` script (which should be automatically run, should install the necessary dependencies)

To test

    ./darknet

You should get the output:

    usage: ./darknet <function>


#### Configuration

1. Set up the config file. For this I've taken a copy of `yolov3-tiny.cfg` (see `cfg/spark-counter-yolov3-tiny.cfg`)

Configure the batch and subdivision. I've used

    batch=64
    subdivisions=16

Set `classes` to 1.

Set `filters` on line 128 and 172 to 18 according to following formula

    filters=(classes + 5)*3

2. Create the names file (listing labels for objects to detect), see `cfg/spark-counter.names`

3. Create `train.txt` and `text.txt`, which list paths to your training and test images.

    find /absolute/path/to/training/ -name "*.jpg" > /path/to/train.txt  
    find /absolute/path/to/testing/ -name "*.jpg" > /path/to/train.txt

4. Create the data file (see `cfg/spark-counter.data`)

    classes= 1
    train  = /path/to/train.txt  
    valid  = /path/to/test.txt  
    names = cfg/counters.names  
    backup = backup

`classes` is the number of classes you'll be detecting. `train`, `valid` and `names` are the paths to the files uploaded in (2) and (3). `backup` is an empty directory you'll need to create and is where trained weights are saved.

5. Generate anchors for your training data

    ./darknet detector calc_anchors /path/to/spark-counter.data -num_of_clusters 6 -width 416 -height 416

And update your `cfg/spark-counter-yolov3-tiny.cfg` file from (1) which these anchors.

6. Optional - Transfer learning: convert pre-trained weights to convolutional layer weights

    ./darknet partial /path/to/cfg/spark-counter-yolov3-tiny.cfg /path/to/pretrained.weights pretrained.conv.11 11
    

#### Running the training

    nohup darknet detector train /path/to/spark-counter.data /path/to/spark-counter-yolov3-tiny.cfg pretrained.conv.11 -dont_show -mjpeg_port 8090 -map > /path/to/darknet.counters.log &

**Note** You will need to download some convolutional weights (e.g. for tiny yolo [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)) - or create your own from pretrained weights, see step 6 above.


You can then check progress by 

    tail -10 /var/log/darknet.counters.log.
    
or 

    grep "avg" /var/log/darknet.counters.log

If you EC2 instance is public then you'll be able to view a graph of the training on port `8090`.


### Training digit recognition (on AWS EC2)

The process for training digit recongition is essentially identical to the above, but for changes in configuration process.

#### Generating the dataset

Your meter images are already split into training, testing and validation. The next step is to apply your counter-detection model to generate images of counter regions, preserving that split.

Either locally in your container or on the EC2 instance run

    python test_counter_detection.py /path/to/dataset/ /path/to/output/

This will find any images in the folder and apply the model (you may need to change the arugments passed to `YoloModel`). It will then generate the following images:

 - `{filename}-prediction.jpg` - the meter image with a bounding box drawn around the predicted counter region (incase of multiple predictions, the one with the highest confidence score is shown)
 - `{filename}-scaled-crop.jpg` - bounding box is scaled up by 20% and the image cropped to that region

The second file forms the dataset for digit recognition training. 


#### Configuration

Configuration follows as before, with the following changes:

 - Add `flip=0` in your config file (`cfg/spark-digits-yolov3-tiny.cfg`) at the top above `learning_rate`. This is to prevent the software from flipping the image while performing data augmentation - this is clearly not suitable when training for digit recognition).

 - Set `classes` to 10

 - Set `filters` on line 128 and 172 to 45 

 - Create names file listing digits 0-9 (see `cfg/spark-digits.names`)

 - Set `classes` to 10 in data file and update path to files listed (see `cfg/spark-digits.data`).

 - Generate a new set of anchors for your new data set


#### Testing your trained model


    python test_digit_detection.py /path/to/dataset/

This will apply your trained model (again you'll need to configure the parameters passed to `YoloModel`) and print out the image filename, the actual reading (according to the annotations) and the predicted value. The last line will print a number between 0 and 1, indicating the percentage of reading correctly extracted.


## Resources

- https://pjreddie.com/darknet/install/ (Installation guide for darknet)
- https://github.com/AlexeyAB/darknet - Implementation of Yolo used
- http://arxiv.org/abs/1506.02640 (YOLO paper, explains some of the configuration parameters)
- https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2 (updated version of below for yolov2)
- https://medium.com/@manivannan_data/how-to-train-yolov2-to-detect-custom-objects-9010df784f36 (contains links to annotateed example datasets)
- https://danielcorcoranssql.wordpress.com/2018/12/24/yolov3-custom-model-training-nfpa-dataset/ Walkthrough of a the same example using NPFA data
- https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e (blog explaining the entire process, uses labelling tool: https://github.com/tzutalin/labelImg)
- https://www.learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/ (explains the configurations, also has link to datatsets, plus code: https://github.com/spmallick/learnopencv)
- https://towardsdatascience.com/tutorial-build-an-object-detection-system-using-yolo-9a930513643a
