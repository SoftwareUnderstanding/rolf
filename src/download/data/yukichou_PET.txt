##初步网络HR-net

![](https://i.imgur.com/ddMm0ba.png)

上图是基本的文件目录，其中experiment包含的时候不同网络的配置文件，可以在命令行后面加上 --cfg filepath 选择，实验超参数则可以在文件里面修改。


---
![](https://i.imgur.com/p3QAn0e.png)

tools里面则是dataset的读取数据集、训练和测试文件，直接运行即可，其中train的循环过程则是在lib/core/function文件中。
需要测试或者输出预测cvs文件则在valid里面修改相关代码，下图所示：

![](https://i.imgur.com/jMQmu4u.png)


---
运行命令：
`CUDA_VISIBLE_DEVICES=3,4 python tools/train.py --cfg experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml`

网络pipline详细看：[HR-Net](https://github.com/HRNet/HRNet-Image-Classification)
````
## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)
