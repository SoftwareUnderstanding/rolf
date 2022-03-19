# PanopticFPN-paddle
paddle version for [PanopticFPN](https://arxiv.org/abs/1901.02446)


0. 本复现基于paddle-seg，安装环境时运行pip install -r requirements.txt，同时paddlepaddle-gpu==2.1.2。
1. 解压模型后将best_model文件夹放置在pfpnnet_cityscapes_b8下，将cityscapes数据集放置在datasets目录下，运行python tools/convert_cityscapes.py得到转化后的cityscapes图片，随后修改run_eval.sh中的模型文件位置并运行run_eval.sh 即可得到最终结果。该结果为单尺度测试结果，复现精度为mIoU:79.5%。
2. log文件和模型文件位于百度网盘
3. 运行环境为AIstudio的4*V100 32G，一张卡4张图，运行40k steps，学习率为0.01。

模型文件和以及log百度网盘位置:
链接：https://pan.baidu.com/s/1MZPLol53e2CryEjeT1a_yg 
提取码：PFPN
