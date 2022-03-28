# paddlex-driver-state-recognition

paddlex-baseddriver face detection and state recognition

•该项目该项目使用PaddleX提供的图像分类模型，在kaggle驾驶员状态检测数据集进行训练；

•训练得到的模型能够区分驾驶员正常驾驶、打电话、喝水、与后座交谈等共10种不同动作，主体内测试集上准确率为0.979；

•最后使用PaddleX将模型量化压缩并部署；

最终效果：

![image](https://github.com/Sharpiless/paddlex-driver-state-recognition/blob/master/QQ%E6%88%AA%E5%9B%BE20200602234019.jpg)

# PaddleX工具简介

PaddleX+是飞桨全流程开发工具，集飞桨核心框架、模型库、工具及组件等深度学习开发所需全部能力于一身，打通深度学习开发全流程，并提供简明易懂的Python API，方便用户根据实际生产需求进行直接调用或二次开发，为开发者提供飞桨全流程开发的最佳实践。目前，该工具代码已开源于GitHub，同时可访问PaddleX在线使用文档，快速查阅使用教程和API文档说明。

PaddleX代码GitHub链接：

https://github.com/PaddlePaddle/PaddleX

PaddleX文档链接：https://paddlex.readthedocs.io/zh_CN/latest/index.html

PaddleX官网链接：https://www.paddlepaddle.org.cn/paddle/paddlex


对于图像分类任务，针对不同的应用场景，PaddleX支持近20种图像分类模型，模型列表可参考PaddleX模型库：https://paddlex.readthedocs.io/zh_CN/latest/appendix/model_zoo.html

# 数据集简介

数据集地址：https://www.kaggle.com/c/state-farm-distracted-driver-detection

该数据集由kaggle提供，共包括十种驾驶员状态共超过2万张图片：

    'c0': ’正常驾驶’,
    'c1': '右手发短信',
    'c2': '右手打电话',   
    'c3': '左手发短信',    
    'c4': '左手打电话',    
    'c5': '调整收音机',    
    'c6': '喝水',    
    'c7': '向后伸手',    
    'c8': '整理头发',    
    'c9': '跟乘客说话'

# MobileNetv3简介：

论文《Searching for MobileNet V3》地址：https://arxiv.org/abs/1804.02767

MobileNetV3则在上两个版本上进行继续改进，其核心思想为：

使用了两个黑科技：NAS 和 NetAdapt 互补搜索技术，其中 NAS 负责搜索网络的模块化结构，NetAdapt 负责微调每一层的 channel 数，从而在延迟和准确性中达到一个平衡；

提出了一个对于移动设备更适用的非线性函数 h−swish[x]；

提出了 MobileNetV3−Large和 MobileNetV3−Small两个新的高效率网络，其中Large版本精度更高，Small版本；

提出了一个新的高效分割（指像素级操作，如语义分割）的解码器（decoderdecoderdecoder）；


定义并训练模型

这里使用PaddleX提供的MobileNetV3模型进行训练，共训练20个epoch，batch大小为32，其中初始学习率为0.01，在第10个和第15个epoch分别变为原来的0.1倍。

model = pdx.cls.MobileNetV3_small_ssld(num_classes=num_classes) 

model.train(num_epochs=20, train_dataset=train_dataset, train_batch_size=32, log_interval_steps=50, eval_dataset=eval_dataset, lr_decay_epochs=[10， 15], save_interval_epochs=1, learning_rate=0.01, save_dir='output/mobilenetv3')


模型评估

这里使用测试集评估模型，最终测试集准确率为0.979。

model = pdx.load_model(save_dir)

model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)

7.进行模型量化部署

进行模型量化并保存量化模型

pdx.slim.export_quant_model(model, eval_dataset, save_dir='./quant_mobilenet')

print('done.')


加载量化模型并进行评估

quant_model = pdx.load_model('./quant_mobilenet')

quant_model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)

参数

model(paddlex.models): paddlex加载的模型

test_dataset(paddlex.dataset): 测试数据集

batch_size(int): 进行前向计算时的批数据大小

batch_num(int): 进行向前计算时批数据数量

save_dir(str): 量化后模型的保存目录

cache_dir(str): 量化过程中的统计数据临时存储目录

这里使用

# 小结

本项目使用PaddleX提供的高层接口，快速、高效地完成了驾驶员状态识别的模型训练和部署。通过Python API方式完成全流程使用或集成，该模型提供全面、灵活、开放的深度学习功能，有更高的定制化空间以及更低门槛的方式快速完成产业模型部署,并提供了应用层的软件和可视化服务。


# 关于作者：

北京理工大学 大二在读

感兴趣的方向为：目标检测、人脸识别、EEG识别等

作者博客地址：https://blog.csdn.net/weixin_44936889

更多资源：

更多PaddleX的应用方法，欢迎访问项目地址：

GitHub: https://github.com/PaddlePaddle/PaddleX

Gitee: https://gitee.com/paddlepaddle/PaddleX

如果您加入官方QQ群，您将遇上大批志同道合的深度学习同学。

飞桨PaddleX技术交流QQ群：1045148026

飞桨官方QQ群：703252161。


如果您想详细了解更多飞桨的相关内容，请参阅以下文档。

官网地址：https://www.paddlepaddle.org.cn

飞桨开源框架项目地址：

GitHub: https://github.com/PaddlePaddle/Paddle

Gitee:  https://gitee.com/paddlepaddle/Paddle
