# 中译英

Neural Machine Translation (Chinese-to-English) based on [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## Requirements

### 环境
- python 3.6
- TensorFlow 1.12.0
- tensor2tensor 1.10.0
- jieba 0.39
- tensorflow_serving_api

### 可选择用docker 来运行
 - 创建container

   `nvidia-docker run -id --name myt2t -v /home/nlp/:/nlp tcxia/tensor2tensor`
 
 - 进入container

   `nvidia-docker exec -it myt2t bash`
 
### 安装相关python包
 切换到项目目录下:
 `pip install -r requirements.txt`

## Prepare Data(数据已经处理好，可跳过)
1. Download the dataset and put the dataset in ***data*** direction
2. Run the data preparation script
    
    `cd /nlp/tcxia/MachineTranslation_t2t/train`
    
    `./self_prepare.sh`
    
3. 处理好的数据保存在 train/t2t_data目录下:
![image](https://github.com/tcxdgit/MachineTranslation_t2t/raw/master/images/t2t_data.PNG)
    
4. 数据文件内容:
![image](https://github.com/tcxdgit/MachineTranslation_t2t/raw/master/images/corpus_zhen.png)
    
## Train Model
训练时，可以通过修改`self_run.sh`脚本中第4行的`export CUDA_VISIBLE_DEVICES=""`来指定GPU.

`self_run.sh`脚本包含两部分， **数据生成** 和 **训练**， 数据生成会按顺序生成vocabulary、lang文件和二进制数据文件(只需生成一次),生成数据过程中程序中断的话，需要把`t2t_data`目录下的二进制格式的文件(形如`translate_zhen_ai-*-*-of-*`)删掉，否则再次运行会报错，这应该是tensor2tensor的一个bug。

`cd /nlp/tcxia/MachineTranslation_t2t/train`

运行训练脚本：
`./self_run.sh` 


## Inference（这里只是离线推理，用来评估模型效果）
Run the inference script

`./self_infer.sh` 

## 在线推理

这里使用 tensorflow serving来提供推理服务

### 导出模型（导出成 tensorflow serving 可以调用的格式, 用来进行在线推理）

可以通过修改`export_model.sh`脚本中第3行的`export CUDA_VISIBLE_DEVICES=""`来指定GPU
`./export_model.sh`

### 启动服务端
需要安装tensorflow serving，[安装方式](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)

 可选择用docker来运行
 - 创建container
 这里要注意映射出服务端口( 端口号可自行设定，这里设置为8502)，如果服务器开了防火墙，记得把端口加入白名单，[设置方式](https://www.cnblogs.com/zl1991/p/10531726.html)
 
   `nvidia-docker run -id --name tf-serving -p 8502:8502 -v /home/nlp/:/nlp ainf-tensorflow-serving-gpu:v1.1`
 
 - 进入container

   `nvidia-docker exec -it tf-serving bash`
  
 - 启动服务端
 
    进入项目train目录
    
   `server.sh`中指定`--port=8502`
  
   启动服务
   `./server.sh`

## 启动客户端，实现翻译功能
   在client.sh脚本中指定你翻译服务的地址和端口, 设置参数`--server=3.2.1.10:8502` 

   在容器myt2t中运行脚本

   `./client.sh`
   
   可以实现翻译:
   
   ![image](https://github.com/tcxdgit/MachineTranslation_t2t/raw/master/images/translation.PNG)

   翻译第一个句子的时候因为要加载jieba词典，所以会慢一些
  
## References

Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Full text available at: https://arxiv.org/abs/1706.03762

Code availabel at: https://github.com/tensorflow/tensor2tensor
