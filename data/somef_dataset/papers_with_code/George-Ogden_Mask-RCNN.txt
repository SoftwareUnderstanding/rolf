# Mask-RCNN Implementation
This is a PyTorch implementation of Mask-RCNN. It uses the webcam or an input video and displays the output or saves the video. See [usage](#usage) for more information.
View the paper at [arxiv.org/abs/1703.06870v1](https://arxiv.org/abs/1703.06870v1) or the PyTorch model source code [pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html).
## Example
See this example of the 100m final on YouTube (which I ran through this code):
[![100m World Record](http://img.youtube.com/vi/fRnFHRNIQRs/0.jpg)](http://www.youtube.com/watch?v=fRnFHRNIQRs "Mask-RCNN Implementation (100m World Record)")<br>
And this is quite a famous frame from a video that I've run it through:
![Example Image](example.jpg "Example")
## Setup
### pip
`pip install -r requirements.txt`
### conda
`conda env create -f env.yaml`
## Usage
```
usage: main.py [-h] [--grey-background] [--classes CLASSES [CLASSES ...]]
               [--detection-threshold DETECTION_THRESHOLD] [--mask-threshold MASK_THRESHOLD]   
               [--max-detections MAX_DETECTIONS]
               [--hide-output | --display-title DISPLAY_TITLE] [--hide-boxes] [--hide-masks]   
               [--hide-labels] [--mask-opacity MASK_OPACITY] [--show-fps]
               [--text-thickness TEXT_THICKNESS] [--box-thickness BOX_THICKNESS]
               {image,folder,video,webcam} ...

Mask-RCNN (segmentation model) implementation in PyTorch

positional arguments:
  {image,folder,video,webcam}

optional arguments:
  -h, --help            show this help message and exit
  --grey-background, -g
                        make the background monochromatic
  --classes CLASSES [CLASSES ...], -c CLASSES [CLASSES ...]
                        limit to certain classes (all or see classes.txt)
  --detection-threshold DETECTION_THRESHOLD
                        confidence threshold for detection (0-1)
  --mask-threshold MASK_THRESHOLD
                        confidence threshold for segmentation mask (0-1)
  --max-detections MAX_DETECTIONS
                        maximum concurrent detections (leave 0 for unlimited)
  --hide-output         do not show output
  --display-title DISPLAY_TITLE
                        window title
  --hide-boxes          do not show bounding boxes
  --hide-masks          do not show segmentation masks
  --hide-labels         do not show labels
  --mask-opacity MASK_OPACITY
                        opacity of segmentation masks
  --show-fps            display processing speed (fps)
  --text-thickness TEXT_THICKNESS
                        thickness of label text
  --box-thickness BOX_THICKNESS
                        thickness of boxes
```
### Image
```
usage: main.py image [-h] --input-image INPUT_IMAGE [--save-path SAVE_PATH | --no-save]

optional arguments:
  -h, --help            show this help message and exit
  --input-image INPUT_IMAGE, --input INPUT_IMAGE, -i INPUT_IMAGE
                        input image
  --save-path SAVE_PATH, --output SAVE_PATH, -o SAVE_PATH
                        output save location
  --no-save             do not save output image
```
### Folder
```
usage: main.py folder [-h] --input-folder INPUT_FOLDER [--output-folder OUTPUT_FOLDER | --no-save] [--extensions EXTENSIONS [EXTENSIONS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --input-folder INPUT_FOLDER, --input INPUT_FOLDER, -i INPUT_FOLDER
                        input folder
  --output-folder OUTPUT_FOLDER, --output OUTPUT_FOLDER, -o OUTPUT_FOLDER
                        output save location
  --no-save             do not save output images
  --extensions EXTENSIONS [EXTENSIONS ...], -e EXTENSIONS [EXTENSIONS ...]
                        image file extensions
```
### Video
```
usage: main.py video [-h] --input-video INPUT_VIDEO [--save-path SAVE_PATH | --no-save]

optional arguments:
  -h, --help            show this help message and exit
  --input-video INPUT_VIDEO, --input INPUT_VIDEO, -i INPUT_VIDEO
                        input video
  --save-path SAVE_PATH, --output SAVE_PATH, -o SAVE_PATH
                        output save location
  --no-save             do not save output video
```
### Webcam
```
usage: main.py webcam [-h] [--source SOURCE] [--save-path SAVE_PATH] [--output-fps OUTPUT_FPS] [--no-save]

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE, --input SOURCE, -i SOURCE
                        webcam number
  --save-path SAVE_PATH, --output SAVE_PATH, -o SAVE_PATH
                        output save location
  --output-fps OUTPUT_FPS
                        output fps for video
  --no-save             do not save output video
```
## Classes
For a list of classes, see [classes.txt](classes.txt). The model has been trained on the COCO dataset so there are 80 classes. **Note: the model outputs 91 classes - one is the background and 10 are `None`**
