# YOLO.JS
A work in progress implementaion of the YOLO object detection in javascript running on top of Tensorflow.js 

[![Build Status](https://hiddentn.visualstudio.com/YOLO.JS/_apis/build/status/TheHidden1.YOLO.JS?branchName=master)](https://hiddentn.visualstudio.com/YOLO.JS/_build/latest?definitionId=3&branchName=master)
[![version](https://img.shields.io/github/package-json/v/TheHidden1/YOLO.JS.svg)](https://github.com/TheHidden1/YOLO.JS/releases)
[![npm](https://img.shields.io/npm/v/@hiddentn/yolo.js.svg)](https://www.npmjs.com/package/@hiddentn/yolo.js)
[![license](https://img.shields.io/github/license/TheHidden1/YOLO.JS.svg)](https://github.com/TheHidden1/YOLO.JS/blob/master/LICENSE)

this Readme is outdated and i will edit it soon

## Examples
#### YOLOv2-Light
![](img/yolo-light-v2.gif)
detections with yolo-v2-light with 416x416 input size on a GTX 1050ti/Chrome/Win10x64  ± 25 FPS :dash:

#### YOLOv3
![](img/yolo-full-v3.gif)
detections with yolo-v3 pretrained weights with 224x224 input size on a GTX 1050ti/Chrome/Win10x64  ± 9 FPS

Video source source: https://www.youtube.com/watch?v=u68EWmtKZw0

## Usage

```cmd
> git clone ... 
> npm install
> webpack
```
if everything went sucessfully, you should see a `yolo.js` in the `/dist` folder

### Detector:eyes:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
<script src="path/to/yolo/yolo.js">
```
```javascript
const config = {
    // Model URL
    modelURL: '',
    // Model version : this is important as there is some post processing changes between yolov2 and yolov3
    // 'v2' ||'v3'
    version: 'v2',
    // this is the size of the model input image : you can lower this to gain more performance
    modelSize: 416,
    // Intersection over inion Threshold and Class probability Threshold
    // we use these to filter the output of the neuralnet
    iouThreshold: 0.5,
    classProbThreshold: 0.5,
    // max detection output
    maxOutput: 20,
    // class labels
    labels: COCO_CLASSES,
    // more info see: https://arxiv.org/pdf/1612.08242.pdf
    anchors: [
        [0.57273, 0.677385],
        [1.87446, 2.06253],
        [3.33843, 5.47434],
        [7.88282, 3.52778],
        [9.77052, 9.16828],
    ],
    masks: [[0, 1, 2, 3, 4]],
    // this is just more customization options concerning the preprocessing phase
    preProcessingOptions: {
        // 'NearestNeighbor'  - this output a more accurate image but but take a bit longer
        // 'Bilinear' - this faster but scrifices image quality
        ResizeOption: 'Bilinear',
        AlignCorners: true,
  },
}
// Or you can use one of the pre made configs but you need to specify the model url yourself //
const config = {
    ...YOLO.tinyYOLOLiteConfig,
    // you can also edit them here
    modelSize: 224,
    modelURL: '',
}

const detector = new YOLO.Detector(config);

detector.load().then(() => {
    detector.detectAsync(img).then((dets) => {
        console.log(dets)
    });
});
// OR 
await detector.load()
const detections = await detector.detectAsync()
console.log(detections)
```

### Classifier:telescope:
WIP

