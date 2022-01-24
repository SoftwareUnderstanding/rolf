# YOLOv3
开发中...(Developing...)敬请期待(Coming soon).   
前馈已经过测试，loss正在开发。(forward has been tested but `loss.py` is developing.)    

2080Ti GPU上, FPS 达50以上. 接近原作者C实现的预测速度.  (可通过`pred_video.py`进行测试)  

## Reference
1. 论文(paper):   
[https://arxiv.org/pdf/1804.02767.pdf](https://arxiv.org/pdf/1804.02767.pdf)  
论文作者实现(Original Implementation):   
[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)  


2. 代码参考(reference code):  
[https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)  
[https://github.com/BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch)  


3. EfficientNet 主干网代码来源(Backbone code source):  
[https://github.com/Jintao-Huang/Darknet53_PyTorch](https://github.com/Jintao-Huang/Darknet53_PyTorch)   

4. 预训练模型来自(The pre-training model comes from):  
[https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)   
从`.weights`模型转成了`.pth`, 并进行了发布


权重见 release. 或在百度云中下载:  
链接：[https://pan.baidu.com/s/1wQ4LQ0yWvLcriqqhSbFoUQ](https://pan.baidu.com/s/1wQ4LQ0yWvLcriqqhSbFoUQ)  
提取码：d6jh   



## 使用方式(How to use)

#### 1. 预测图片(Predict images)
```
python3 pred_image.py
```

#### 2. 预测视频(Predict video)
```
python3 pred_video.py
```

#### 3. 简单的训练案例(Simple training cases)
```
python3 easy_examples.py
```

## 网络架构图
如果打不开可在`images/`与`docs/`文件夹中查看  

![网络架构图](./docs/YOLOv3网络架构图.png)

#### 效果

![原图片](./images/1.png)

![检测图片](./images/1_out.png)


## 运行环境(environment)

torch 1.2.0  
torchvision 0.4.0  
