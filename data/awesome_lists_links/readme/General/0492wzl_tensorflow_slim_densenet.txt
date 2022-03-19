# 简介
本代码为系列课程, 第七周部分的课后作业内容。
https://edu.csdn.net/topic/ai115


学员自己实现一个densenet的网络，并插入到slim框架中进行训练。

tinymind 使用说明：https://gitee.com/ai100/quiz-w7-doc


### 数据集
本数据集拥有200个分类，每个分类300张图片，共计6W张图片，其中5W张作为训练集，1W张图片作为验证集。图片已经预打包为tfrecord格式并上传到tinymind上。地址如下：
https://www.tinymind.com/ai100/datasets/quiz-w7

### 模型
模型代码来自：
https://github.com/tensorflow/models/tree/master/research/slim


这里为了适应本作业提供的数据集，稍作修改，添加了一个quiz数据集以及一个训练并验证的脚本，实际使用的代码为：
https://gitee.com/ai100/quiz-w7-2-densenet


其中nets目录下的densenet.py中已经定义了densenet网络的入口函数等，相应的辅助代码也都已经完成，学员只需要check或者fork这里的代码，添加自己的densenet实现并在tinymind上建立相应的模型即可。


densenet论文参考 https://arxiv.org/abs/1608.06993


在tinymind上新建一个模型，模型设置参考如下模型：

https://www.tinymind.com/ai100/quiz-w7-2-densenet
复制模型后可以看到模型的全部参数。

模型参数的解释：

- dataset_name quiz  # 数据集的名称，这里使用我们为本次作业专门做的quiz数据集
- dataset_dir /data/ai100/quiz-w7  # tfrecord存放的目录，这个目录是建立模型的时候，由tinymind提供的
- model_name densenet  # 使用的网络的名称，本作业固定为densenet
- train_dir /output/ckpt  # 训练目录，训练的中间文件和summary，checkpoint等都存放在这里，这个目录也是验证过程的checkpoint_path参数， 这个目录由tinymind提供，需要注意这个目录是需要写入的，使用其他目录可能会出现写入失败的情况。
- learning_rate 0.1  # 学习率, 因为没有预训练模型，这里使用较大的学习率以加快收敛速度。
- optimizer rmsprop  # 优化器，关于优化器的区别请参考[这里](https://arxiv.org/abs/1609.04747)
- dataset_split_name validation # 数据集分块名，用于验证过程，传入train可验证train集准确度，传入validation可验证validation集准确度，这里只关注validation
- eval_dir /output/eval  # 验证目录，验证结果，包括summary等，会写入这个目录
- max_num_batches 128  # 验证batches，这里会验证128×32共4096个图片样本的数据。


鼓励参与课程的学员尝试不同的参数组合以体验不同的参数对训练准确率和收敛速度的影响。

### 结果评估
学员需要提供运行log的截图和文档描述

在tinymind运行log的输出中，可以看到如下内容：
```sh
2017-12-1 23:03:04.327009: I tensorflow/core/kernels/logging_ops.cc:79] eval/Accuracy[0.252197266]
2017-12-1 23:03:04.327097: I tensorflow/core/kernels/logging_ops.cc:79] eval/Recall_5[0.494873047]
```
densenet的网络，效果要略好于inceptionv4。考虑到实现的不同，而且没有预训练模型，这里不对准确率做要求。只要训练运行成功并有准确率输出即可认为及格60分。

提供对densenet实现过程的描述：
对growth的理解 20分
对稠密链接的理解 20分


# 参考内容
>epoch计算方式：
>epoch = step * batch_size / count_all_train_pics


本地运行slim框架所用命令行：

使用预训练模型进行inceptionv4等的finetune
```sh
训练：
python3 train_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/path/to/train_ckpt --learning_rate=0.001 --optimizer=rmsprop  --batch_size=32

train集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=train --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/train_eval --batch_size=32 --max_num_batches=128

validation集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=validation --model_name=inception_v4 --checkpoint_path=/path/to/train_ckpt --eval_dir=/path/to/validation_eval --batch_size=32 --max_num_batches=128

统一脚本：
python3 train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --checkpoint_path=/path/to/inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --optimizer=rmsprop --train_dir=/path/to/log/train_ckpt --learning_rate=0.001 --dataset_split_name=validation --eval_dir=/path/to/eval --max_num_batches=128
```

从头开始训练densenet
```sh
训练
python3 train_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --model_name=densenet --train_dir=/path/to/train_ckpt_den --learning_rate=0.1 --optimizer=rmsprop  --batch_size=16/path/to

train集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=train --model_name=densenet --checkpoint_path=/path/to/train_ckpt_den --eval_dir=/path/to/train_eval_den --batch_size=32 --max_num_batches=128

validation集验证：
python3 eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --dataset_split_name=validation --model_name=densenet --checkpoint_path=/path/to/train_ckpt_den --eval_dir=/path/to/validation_eval_den --batch_size=32 --max_num_batches=128

统一脚本：
python3 train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=/path/to/data --model_name=densenet --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/path/to/log/train_ckpt --learning_rate=0.1 --dataset_split_name=validation --eval_dir=/path/to/eval_den --max_num_batches=128
```

## cpu训练
本地没有显卡的情况下，使用上述命令进行训练会导致错误。只使用CPU进行训练的话，需要在训练命令或者统一脚本上添加**--clone_on_cpu=True**参数。tinymind上则需要新建一个**clone_on_cpu**的**bool**类型参数并设置为**True**

