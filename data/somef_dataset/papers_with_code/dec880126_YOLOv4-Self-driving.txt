<h1 align="center">Self-driving training with YOLO</h1>

* [Demo](#Demo)
* [Dataset](#dataset)
* [AP](AP)
* [mAP](#mAP)
* [Classes](#Classes)
* [Training Log](#Log)
* [Weights](#weights)
* [How TO Use](#how)
* [Environment](#environment)
* [Speed](#speed)
* [YouTube Link of Video Demo](#ytlink)
* [Google Colab](#colab)
* [Tutorial](#tutorial)
  * [Run your custom object detection](#run)
* [Citation](#citation)
* [References](#references)

<h2 id='Demo'>Demo</h2>

YOLOv4             |  YOLOv3
:-------------------------:|:-------------------------:
![1627216237257](https://user-images.githubusercontent.com/34447298/126899013-21aadbf8-b79e-46ad-b63e-0d3d0f59cab5.gif) | ![1627544679170](https://user-images.githubusercontent.com/34447298/127452341-4deb4463-1c65-4923-a30f-6aa8326c46cd.gif)

<h2 id='dataset'>Dataset</h2>

![截圖 2021-08-03 21 18 55](https://user-images.githubusercontent.com/34447298/128022232-d4b5ebd2-82bb-4ac4-a388-b48c02ad3baf.png)


<h2 id='AP'>AP</h2>

![截圖 2021-08-03 03 45 54](https://user-images.githubusercontent.com/34447298/127915198-ae6b5b4c-1f02-4e89-8e28-6cd49f6b5892.png)

class             |  AP in YOLOv4 |  AP in YOLOv3 | TP&FP in YOLOv4 | TP&FP in YOLOv3
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
car | ap = 73.09% | ap = 69.30% | TP = 15977, FP = 5767 | TP = 15037, FP = 6829
truck | ap = 61.61% | ap = 51.89% | TP = 573, FP = 232 | TP = 469, FP = 244
pedestrian | ap = 42.53% | ap = 24.20% | TP = 2192, FP = 1392 | TP = 1213, FP = 1242
bicyclist | ap = 41.32% | ap = 15.66% | TP = 93, FP = 63| TP = 51, FP = 94
light | ap = 51.58% | ap = 42.93% | TP = 2298, FP = 739 | TP = 1793, FP = 706
<br>

Conclusion: **More significant improvement in low AP classes.**

<h2 id='mAP'>mAP</h2>

![截圖 2021-08-03 03 42 48](https://user-images.githubusercontent.com/34447298/127914825-0ee3bd22-cab0-42a9-9d09-321ed0dee082.png)

> for 10,000 iterations

YOLOv4             |  YOLOv3
:-------------------------:|:-------------------------:
**mean average precision (mAP@0.50) = 54.02 %** | **mean average precision (mAP@0.50) = 40.80 %**


<h2 id='Classes'>Classes</h2>

![截圖 2021-08-03 21 15 21](https://user-images.githubusercontent.com/34447298/128021722-e45a35d9-6b51-402a-92fa-46c1f39284d2.png)

1. car: with 101314 labels
2. truck: with 6313 labels
3. pedestrian: with 10637 labels
4. bicyclist: with 1442 labels
5. light: with 12700 labels

<h2 id='Log'>Training Log</h2>

YOLOv4             |  YOLOv3
:-------------------------:|:-------------------------:
![chart 2](https://user-images.githubusercontent.com/34447298/127896489-c0760257-baf9-4b7b-b9de-a7ec24c86907.jpg)| ![chart](https://user-images.githubusercontent.com/34447298/127447409-ed86928f-d060-440b-925c-fc0bedb69b0c.png)

Conclusion: **The speed of convergence in YOLOv4 is much faster than that in YOLOv3**

<h2 id='weights'>Weights</h2>

YOLOv4             |  YOLOv3
:-------------------------:|:-------------------------:
<a href='https://drive.google.com/file/d/1UcwVXnIwra52eKY-a8jHg2AT6tALBnpw/view?usp=sharing' target="_blank">yolov4-obj_10000.weights</a> | <a href='https://drive.google.com/file/d/1Hrf_RzsQWD8oRv5UX37C9JvF8QO8w7qp/view?usp=sharing' target="_blank">yolov3-obj_10000.weights</a>


<h2 id='how'>How To Use</h2>

- Use with <a href='https://github.com/AlexeyAB/darknet' target="_blank">YOLOv4 AlexeyAB</a>


<h2 id='environment'>Environment</h2>

- VM: Google Colaboratory
- GPU: NVIDIA T4 Tensor GPU
- NVIDIA-SMI 470.42.01    Driver Version: 460.32.03    CUDA Version: 11.2
- nvcc: NVIDIA (R) Cuda compiler driver
- Cuda compilation tools, release 11.0, V11.0.221
- Build cuda_11.0_bu.TC445_37.28845127_0


<h2 id='speed'>Speed</h2>

![截圖 2021-08-03 03 37 19](https://user-images.githubusercontent.com/34447298/127914264-eae20dd4-0a38-4b0c-92ff-c774a1191a66.png)

| | YOLOv4             |  YOLOv3
|:-------------------------:|:-------------------------:|:-------------------------:
|Quality: 1080p | AVG FPS = 14.6 | AVG FPS = 16.1
|Quality: 720p | AVG FPS = 33.1 | AVG FPS = 33.0
|Quality: 360p | AVG FPS = 45.2 | AVG FPS = 43.4
| mAP | 54.02 % | 40.80 %


<h2 id='ytlink'>YouTube Link of Video Demo</h2>

| | YOLOv4             |  YOLOv3
|:-------------------------:|:-------------------------:|:-------------------------:
|Quality: 1080p | <a href='https://youtu.be/z7_G99y6Tj0'>Click Me</a> | <a href='https://youtu.be/Ui31LWzBznY'>Click Me</a>
|Quality: 720p | <a href='https://youtu.be/FCZUgSq0MC4'>Click Me</a> | <a href='https://youtu.be/tIrTPZjyQCI'>Click Me</a>
|Quality: 360p | <a href='https://youtu.be/dCZqnxwqpzs'>Click Me</a> | <a href='https://youtu.be/X-daaeXujCQ'>Click Me</a>
| mAP | 54.02 % | 40.80 %
<br>
<a href='https://youtu.be/8aLlVEkXN5Q'>Video of comparison between YOLOv4 & YOLOv3</a>

<h2 id="colab">Google Colab</h2>

<a href="https://colab.research.google.com/drive/1nPpz8-5tg6nwoMtCYWgunGIlMM8j0yui?usp=sharing">Link</a>

<h2 id="tutorial">Tutorial</h2>

### 檢查環境用指令
1. verify CUDA version: `/usr/local/cuda/bin/nvcc --version`
2. check GPU info: `nvidia-smi`

### 配置Darknet環境
> clone AlexeyAB/darknet repo

`git clone https://github.com/AlexeyAB/darknet`

> change makefile to have GPU and OPENCV enabled

```
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
```

> Build darknet environment

`make`

> 配置config檔

 - change line batch to `batch=64`
 - change line subdivisions to `subdivisions=16`
 - change line max_batches to (`classes*2000`, but not less than number of training images and not less than `6000`), f.e. max_batches=6000 if you train for 3 classes
 - change line steps to 80% and 90% of max_batches, f.e. `steps=4800,5400`
 - set network size `width=416 height=416` or any value multiple of 32:
 - change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers
 - change `[filters=255]` to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers

> 準備好以下檔案

1. train.txt
2. test.txt
3. obj.data
4. obj.names
5. pre-trained.weights

### Start training

`./darknet detector train <your_path_of_obj.data> <your_path_of_cfg> <your_path_of_weights> -chart chart.png`

 - `-chart chart.png`: 可保存訓練過程

<h2 id="run">Run your custom object detection</h2>

### 修改cfg
```
sed -i 's/batch=64/batch=1/' <your_path_of_cfg>
sed -i 's/subdivisions=16/subdivisions=1/' <your_path_of_cfg>
```

### Detect

`./darknet detector test <your_path_of_obj.data> <your_path_of_cfg> <your_path_of_weights> <your_path_of_input_picture>`

### 計算mAP

`./darknet detector map <your_path_of_obj.data> <your_path_of_cfg> <your_path_of_weights>`

<h2 id='citation'>Citation</h2>

```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```

<h2 id='ref'>References</h2>

- YOLOv4: Optimal Speed and Accuracy of Object Detection: https://arxiv.org/pdf/2004.10934
- Training data from: https://www.kaggle.com/alincijov/self-driving-cars
- Tesing data from: https://youtu.be/z1obnaqPgMA
