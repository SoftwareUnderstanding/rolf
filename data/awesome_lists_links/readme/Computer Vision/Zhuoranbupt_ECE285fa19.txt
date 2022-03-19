ECE 285 MLIP Project C Multi-Object detection using Deep Learning
===========================
This is a simplified introduction of ECE 285 project--Multi-Object detection.

****
	
|Author|
|---
|Fei Xue|
|Bo Zhou|
|Zhuoran Liu|

## Invironment configuration
---

```

git clone https://github.com/Zhuoranbupt/ECE285fa19/tree/master
pip install --user scipy == 1.3.0
pip install --uer scikit-image

```
To run the demo, we need to download pretrained weights
```
cd weights/
bash download_weights.sh
```
Then fetch the yolov3.pth from https://drive.google.com/open?id=1W9_TWsQ25GjqBA5f_F17L9LjOnq_xFRX and put it into checkpoints

## Dataset
---
The dataset we used in the project is PascalVOC2012, the JPEGImages for images and Annotations for labels.

```
ln -s /datasets/ee285f-public/PascalVOC2012 
```

## Code organization
---
```
augmentations.py -- establishes measurement of the network
dataloder.py -- imports the data from the file and outputs image-label pairs
datasets.py -- establishes measurement of the network
demo.ipynb -- deploys a detection demo of the model
models.py -- contains setting of the model
nntols.py -- contains setting of the network
parse_config.py -- reads the config files of YOLOv3 
test.py -- establishes measurement of the network
utils.py -- contains information used in model training
yolov3.ipynb -- a program that consists of main functions of the project
```


## Train
---
The loss and mAP line charts showed the model opitimizing during the 100 epochs. 

![loss.png](https://github.com/Zhuoranbupt/ECE285fa19/blob/master/result_image/loss.png?raw=true)

Loss changes of the model

![mAP.png](https://github.com/Zhuoranbupt/ECE285fa19/blob/master/result_image/mAP.png?raw=true)

mAP changes of the model
## Test
---
We tested the model on two pictures. The first picture contains a car and two people. The result was that the model detected the car and one person since the other person is to small to be detected.

![test1.png](https://github.com/Zhuoranbupt/ECE285fa19/blob/master/result_image/test1.png?raw=true)

Another test was detecting a computer. The detecting box was accurate and the label was correct.

![test2.png](https://github.com/Zhuoranbupt/ECE285fa19/blob/master/result_image/test2.png?raw=true)


## Acknowledgement
---
### YOLOv3: An Incremental Improvement
```
@article{DBLP:journals/corr/abs-1804-02767,
  author    = {Joseph Redmon and
               Ali Farhadi},
  title     = {YOLOv3: An Incremental Improvement},
  journal   = {CoRR},
  volume    = {abs/1804.02767},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.02767},
  archivePrefix = {arXiv},
  eprint    = {1804.02767},
  timestamp = {Mon, 13 Aug 2018 16:48:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-02767},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.
https://pjreddie.com/media/files/papers/YOLOv3.pdf

