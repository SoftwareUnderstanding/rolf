# COVID-19-Mask_detector
## Real Time Mask Detection

In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. 

## Tech/framework used

- [OpenCV](https://opencv.org/)
- [Darknet](https://github.com/SravanChittupalli/darknet)

## YOLO Architecture
Paper Yolo v4: https://arxiv.org/abs/2004.10934

![Yolo Architecture](https://miro.medium.com/max/2864/1*b-TbPh9J0Oyal7Nw6iah5w.jpeg)

YOLO is a clever convolutional neural network (CNN) for doing object detection in real-time. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region.

## STEPS
1. Clone this repository
2. Clone [AlexyAB's](https://github.com/SravanChittupalli/darknet.git) darknet implementation
3. Follow AlexyAB's README to build the repo. Run the demo successfully
4. Copy this repo's `mask.py` and paste it into `./darknet/`
5. Copy this repo's `yolo-obj.cfg` and paste it into `./darknet/cfg/`
6. Copy this repo's `obj.data` and paste it into `./darknet/cfg`
7. Copy this repo's `obj.name` and paste it into `./darknet/data/`
8. You can find pre-trained weights [here](https://drive.google.com/drive/folders/14LGXxTuqg3bg6rIVvPTyKWU2vBZT5ZYe?usp=sharing)<br>
There are 2 weight files one is using Roboflow's mask dataset that is good for mask detection in cloase range and the other one is a custom dataset made by  [Aditya Purohit](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format/metadata) to which I have added my own data.

## Train
Follow the [notebook](https://github.com/SravanChittupalli/COVID-19-Mask_detector/blob/master/training_yolov4.ipynb) that is included in the repository.

## Dataset 
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/1otjrAa3UCdvTFiqRnUo2i1uw7v0B-Hao?usp=sharing)


## Video of results
[![Click here for demo video](https://github.com/SravanChittupalli/COVID-19-Mask_detector/blob/master/Mask_demo.png)](https://youtu.be/U1RNoMJoy_Y)

## Acknowledgements
  - [Roboflow's Mask Wearing Dataset](https://public.roboflow.ai/object-detection/mask-wearing/1)
  - [Aditya Purohit Crowd dataset](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format/metadata)

## Tutorials
Roboflow :- [How to Train YOLOv4 on a Custom Dataset](https://blog.roboflow.ai/training-yolov4-on-a-custom-dataset/)
