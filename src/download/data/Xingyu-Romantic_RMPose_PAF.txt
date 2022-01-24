

# RMPose_PAF





Ai Studio: [https://aistudio.baidu.com/aistudio/projectdetail/2306743](https://aistudio.baidu.com/aistudio/projectdetail/2306743)

《Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ∗ 》论文复现

训练权重：[https://drive.google.com/file/d/1Fz1MP4ybmbhr1oP2dPRFj0T54YUkmKNZ/view?usp=sharing](https://drive.google.com/file/d/1Fz1MP4ybmbhr1oP2dPRFj0T54YUkmKNZ/view?usp=sharing)

VGG19训练权重：[https://drive.google.com/file/d/1xZJyh-scsvq4bdHdRgs5Ou-L0qtdx9JN/view?usp=sharing](https://drive.google.com/file/d/1xZJyh-scsvq4bdHdRgs5Ou-L0qtdx9JN/view?usp=sharing)

下载后放在`./`目录下

## Structure

文件结构

├── data
│   └── mpii
│       ├── annot
│       └── images
└── RMPose_PAFs
    ├── ...



`pose_estimation.py`:   `Model `

`ski.jpg` : Test image

`nohup.out`: Train Log



------

# 模型整体流程

## 首要条件
```python
export PYTHONPATH="$PWD":$PYTHONPATH  #终端执行
```

## 训练
可通过`training/config.yml` 文件夹修改训练超参数
```python
python training/train_pose.py --config ./trainning/config.yml --train_dir ./datasets/process_train.json --val_dir ./datasets/process_val.json
```
## 评估
* `model`：模型权重
* `only_eval`: 已有评估完成存在的predection.npy文件，对此评估。

```python
python testing/eval.py  --model ./RMPose_PAFs.pdparams.tar -only_eval True
```

## 测试
* `image_dir`: 支持文件夹路径，以及单文件处理
* `model`: 模型权重
* `output`: 输出文件夹路径
```python
python testing/test_pose.py --image_dir ./ski.jpg --model ./RMPose_PAFs.pdparams.tar
```

## Question

Q: **ModuleNotFoundError**: No module named 'pafprocess_mpi'

A: `export PYTHONPATH="$PWD":$PYTHONPATH`



# 评估指标
参考代码（Matlab版）： [https://github.com/anibali/eval-mpii-pose](https://github.com/anibali/eval-mpii-pose)

MPII数据集的评估指标采用的是PCKh@0.5。预测的关节点与其对应的真实关节点之间的归一化距离小于设定阈值，则认为关节点被正确预测，PCK即通过这种方法正确预测的关节点比例。

PCK@0.2表示以躯干直径作为参考，如果归一化后的距离大于阈值0.2，则认为预测正确。

PCKh@0.5表示以头部长度作为参考，如果归一化后的距离大于阈值0.5，则认为预测正确。

在本项目中论文复现结果：

| Method          | Head  | Shoulder | Elbow | Wrist | Hip   | Knee  | Ankle | Mean  |
| --------------- | ----- | -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| **RMPose_PAFs** | 36.94 | 92.19    | 86.21 | 79.77 | 85.62 | 81.62 | 76.64 | 80.96 |
| **原论文**      | 91.2  | 887.6    | 77.7  | 66.8  | 75.4  | 68.9  | 61.7  | 75.6  |
| **DeeperCut**   | 73.4  | 71.8     | 57.9  | 39.9  | 56.7  | 44.0  | 32.0  | 54.1  |
| **AlphaPose**   | 91.3  | 90.5     | 84.0  | 76.4  | 80.3  | 79.9  | 72.4  | 82.1  |
