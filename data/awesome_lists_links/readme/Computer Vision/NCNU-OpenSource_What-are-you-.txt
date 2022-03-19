## What-are-you?
Group 6

Traditionally, we use flash cards to teach children. We need to stay with them all the time to teach them the basic. However, we are very busy in this society and we can't answer them all the time when they are interesting on what the thing is.
Therefore, childrens can use WHAT-ARE-YOU to learn knowledge and correct their wrong answer. They can take a picture of the thing and WHAT-ARE-YOU would answer it.


+ 使用Mxnet做物體辨識，外加上SSD技術讓辨識可以在一張照片上辨識出多種物品
---
### 什麽是Mxnet？
+ MXNet是一個開源深度學習軟體框架，用於訓練及部署深度神經網路。MXNet具有可延伸性，允許快速模型訓練，
並支援靈活的編程模型和多種程式語言（包括C++、Python、Julia、Matlab、JavaScript、Go、R、Scala、Perl和Wolfram語言）。

### 什麽是SSD?
+ SSD 即 Single Shot Detector
+ 最近一年比较优秀的object detection算法，主要特点在于采用了特征融合。
+ 一种直接预测bounding box的坐标和类别的object detection算法，没有生成proposal的过程
+ 架構圖
<img width="558" alt="ssd" src="https://user-images.githubusercontent.com/35098279/59819988-c4d3cd00-935c-11e9-90a0-b9a67f18e9fc.PNG">

### Demo 
#### Environment
Ubuntu 16.04 (Dual OS with Windows)
GPU： GTX1060 3GB




安裝套件
+ `sudo apt-get install python-opencv python-matplotlib python-numpy`
+ `sudo apt-get install mxnet`
  + 如果你有獨顯
  + `sudo apt-get install mxnet-cu92`
  + https://mxnet.incubator.apache.org/versions/master/install/ubuntu_setup.html#cuda-dependencies
  + 版本要對哦！

+ 下載 pretrained models
  + <a href=https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip>pretrained model</a>
  + 解壓縮放在 data/ 裏面
  
+ 執行demo 
  + `python demo.py` (default settings)
  + `python demo.py --images ./data/demo/dog.jpg --thresh 0.5`
  + `python demo.py --cpu --network resnet50 --data-shape 512`

#### Training
**need to define the epoch by yourself**
1) learning rate: 0.004
+ 1st layer
<img src="https://user-images.githubusercontent.com/29758852/59841410-dc27b000-9386-11e9-9516-982fc7dc2fa2.jpg">
+ 1st layer result
<img src="https://user-images.githubusercontent.com/29758852/59841412-dcc04680-9386-11e9-8976-405dffb91519.jpg">
+ 22nd layer
<img src="https://user-images.githubusercontent.com/29758852/59841413-dcc04680-9386-11e9-95c9-e19784bb266e.jpg">

2) learning rate: 0.001
<img src="https://user-images.githubusercontent.com/29758852/59961492-0013fe80-950b-11e9-917e-5ab89c9b45e3.png">

3) learning rate: 0.001; end-epoch: 17
<img src="https://user-images.githubusercontent.com/29758852/59961491-0013fe80-950b-11e9-9bd8-f8659a4d9208.png">



#### Difficulties
+ install ubuntu as dual OSs with windows

  the setting of bios must be careful.
  1)  secure boot -> disable
  2)  fast boot -> disable
+ error with im2rec.py

  can't fix the orignal code `python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst`
  
  maybe not the problem of opencv(most of the answer on internet said the reason was that there are one and more OpenCV versions but we only have the v2.4.0.1) but cant't find the exact reason and we used 'gdb' to find the error but no useful message found
  
  use another code `python /home/<usrname>/mxnet-ssd/tools/../mxnet/tools/im2rec.py /home/<usrname>/mxnet-ssd/data/train.lst /home/<usrname>/mxnet-ssd/data/VOCdevkit --shuffle 1 --pack-label 1`
+ batch size made the GPU out of memory

  because our GPU has 3GB only so we need to change smaller batch size


#### 可辨識的物品
+ Aeroplane
+ bicycle
+ bird
+ boat
+ bottle
+ bus
+ car
+ cat
+ chair
+ cow
+ diningtable
+ dog
+ horse
+ motorbike
+ person
+ pottedplant
+ sheep
+ sofa
+ train 
+ tvmonitor

## workload
|成員| 貢獻 |
| --------| -------- |
| Dorothy |處理Windows的部分，處理Readme，|
| Jasmine |處理Ubuntu的部分，跑Training|

### 參考資料
+ https://blog.csdn.net/u014380165/article/details/72824889
+ https://blog.csdn.net/u014380165/article/details/78219584
+ https://github.com/zhreshold/mxnet-ssd
+ https://www.cnblogs.com/visiontony/p/how-to-debug-python-segmentation-fault-using-gdb.html
+ https://github.com/tensorflow/models/issues/2034
+ https://paperswithcode.com/paper/ssd-single-shot-multibox-detector (paper)
+ https://arxiv.org/pdf/1512.02325v5.pdf
