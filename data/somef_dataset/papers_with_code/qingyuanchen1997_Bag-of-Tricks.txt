# 代码说明

基于python 3.7.6和pytorch 1.5.0对论文“Bag of Tricks for Image Classification with Convolutional Neural Networks”进行复现，验证了论文中的大部分技巧

### 下载链接
论文地址：
https://arxiv.org/pdf/1812.01187.pdf

猫狗大战数据集下载链接：
链接：https://pan.baidu.com/s/1bRpXfd4ct1Um5qvwYqCu6Q
提取码：s96i

### 文件结构

--project

        --Cat_Dog

                --kaggle

                        --my_test

                        --my_train

                        --teacher_train （给教师模型更多的训练图片）

        --CODE

                --checkpoints（存放模型参数）

                --all .py


### 训练
```python
cd CODE
python3 train.py --ResNet_BCD True  --float16 True  --cosine_decay True  --batch_size 128  --smoothing_label True
#知识蒸馏
python3 teacher.py  #训练教师模型
python3 teacher_label.py  #产生教师标签
python3 learn_from_teacher.py  #训练学生模型
```

### 结果
| 学习率 | 大BS+半精度 | ResNet B+C+D+余弦衰减 | 标签平滑 | 知识蒸馏 | 显存占用 | ACC |
| -- | -- | -- | -- | -- | -- | -- |
| 0.0025 | BS=32 |  |  |  | 2.2G | 0.7425 |
| 0.01 | BS=128 √ |  |  |  | 2.2G | 0.7425 |
| 0.01 | BS=128 √ | √ |  |  | 2.2G | 0.7650 |
| 0.01 | BS=128 √ | √ | √ |  | 2.2G | 0.7700 |
| 0.01 | BS=128 √ | √ |  | √ | 2.2G | 0.7700 |


