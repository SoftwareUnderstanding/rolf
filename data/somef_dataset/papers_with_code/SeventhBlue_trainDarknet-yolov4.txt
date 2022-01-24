![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Yolo v4 paper: https://arxiv.org/abs/2004.10934

Yolo v4 source code: https://github.com/AlexeyAB/darknet

Yolov v4 tiny discussion: https://www.reddit.com/r/MachineLearning/comments/hu7lyt/p_yolov4tiny_speed_1770_fps_tensorrtbatch4/

Useful links: https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


# 一、编译：

	
	1.1 安装opencv：https://blog.csdn.net/qq_31112205/article/details/105161419	

	1.2 根据电脑配置设置Makefile：这里默认使用gpu，cudnn，opencv，CUDNN_HALF，LIBSO
	
	[以/trainDarknet-yolov4为根目录]
	
	1.3 make

# 二、测试：

	[以/trainDarknet-yolov4为根目录]
	
	2.1 下载好权重，yolov4.weights放置到./;yolov4.conv.137(训练时使用)放置到./myData/weights/preWeights/
	
	2.2 测试命令：./darknet detect cfg/yolov4.cfg yolov4.weights data/dog.jpg

		或者 ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
	
	2.3 安装好了opencv会显示检测结果

# 三、训练(请看下面的更新部分)：

	注意：1.在myData/目录下，需要确保myData/Annotations，myData/ImageSets/Main，myData/JPEGImages，myData/weights文件夹存在
	
	      2.根据自己电脑的性能修改batch和subdivisions值，一般来讲电脑性能越好batch值越大，subdivisions值越小，最好是2的整数指数值
	
	3.1 voc数据格式：xml文件放到myData/Annotations，图片放到myData/JPEGImages（文件夹不存在自己新建就行）
	
	3.2 myData.names根据自己的类别填写，要写全

	[以/myData为运行根目录]
	
	3.3 执行命令：python dataPrecess.py
	
	[以/trainDarknet-yolov4为根目录]
	
	3.4 训练执行命令：./darknet detector train myData/cfg/myData.data myData/cfg/myYolov4.cfg myData/weights/preWeights/yolov4.conv.137 -gpus 0 -map

# 四、权重：

	链接：https://pan.baidu.com/s/1Dw3-T9fxcPSbmrXH09u6tQ 
	
	提取码：vgib
	
# 五、训练(更新20201102):
	
	注意：1.在myData/目录下，需要确保myData/Annotations，myData/ImageSets/Main，myData/JPEGImages，myData/weights文件夹存在，如果不存在那么新建该文件夹；同时删除无关文件，文件名上已经标明
	
	      2.根据自己电脑的性能修改batch和subdivisions值，一般来讲电脑性能越好batch值越大，subdivisions值越小，最好是2的整数指数值
	
	3.1 voc数据格式：xml文件放到myData/Annotations，图片放到myData/JPEGImages

	[以/myData为运行根目录]
	
	3.2 执行命令：python dataPrecess.py
	
	[以/trainDarknet-yolov4为根目录]
	
	3.3 训练执行命令：./darknet detector train myData/cfg/myData.data myData/cfg/myYolov4.cfg myData/weights/preWeights/yolov4.conv.137 -gpus 0 -map
