# SimpleRecurrentUnits-SRU-

SRU based acoustic model for merlin
---------------
SRU introduction
>Tao Lei《Simple Recurrent Units for Highly Parallelizable Recurrence》 https://arxiv.org/pdf/1709.02755.pdf

SRU is applied in acoustic model by
DabiaoMa ZhibaSu WenxuanWang ChengZou YuhaoLu @ Turing Robot 

data_preprocess

>file folder for cmp( acoustic feature )and  label (input text label, see HTS label)

data_train

>file folder for model and training files.

基于SRU的声学模型 for merlin
-----------------
马达标 苏志霸 王文轩 邹城 陆羽皓 @图灵机器人

data_preprocess

>放置训练用的特征(cmp文件)和label(lab文件)以及对应的数据索引文件(list)。

data_train

>训练的代码，包括数据载入函数，训练文件和模型定义文件，保存每一个epoch训练出来的trainer文件,以及从trainer提取出model的代码。
