# Paddle-PointNet

_Use PaddlePaddle to implement PointNet (Classifier Only)_


## 1. Introduction

This project reproduces PointNet based on paddlepaddle framework.

PointNet provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.

**Paper:** [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

**Competition Page:** [PaddlePaddle AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/106)

**PointNet Architecture:**
![arch](arch.png)

**Other Version Implementation:**

- [TensorFlow (Official)](https://github.com/charlesq34/pointnet)
- [PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

**Acceptance condition**

- Classification Accuracy 89.2 on ModelNet40 Dataset

## 2. Accuracy

Classification result on ModelNet40

| Model                   | Accuracy |
| ----------------------- | -------- |
| PointNet (Official)     | 89.2     |
| PointNet (PyTorch)      | 90.6     |
| PointNet (PaddlePaddle) | 89.4     |

## 3. Dataset

### [ModelNet40](https://modelnet.cs.princeton.edu)

> The goal of Princeton ModelNet project is to provide researchers in computer vision, computer graphics, robotics and cognitive science, with a comprehensive clean collection of 3D CAD models for objects.

- Dataset size:
  - Train: 9843
  - Test: 2468
- Dataset format:
  - CAD models in [Object File Format](https://segeval.cs.princeton.edu/public/off_format.html)

## 4. Environment

- Hardware: GPU/CPU
- Framework:
  - PaddlePaddle >= 2.1.2

## 5. Quick Start

### Data Preparation

Download [alignment ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `modelnet40_normal_resampled/`. The same dataset as the PyTorch version implementation.

```
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
```

### Train

```
python train.py
```

The model will be saved as `pointnet.pdparams` by default.

### Test

```
python test.py
```

## 6. Details

### Project Structure
```
├── README.md
├── arch.png
├── data.py
├── model.py
├── pointnet.pdparams
├── requirements.txt
├── test.py
├── train.log
└── train.py
```

### T-Net Layer
```python
class TNet(nn.Layer):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k
        self.iden = paddle.eye(self.k, self.k, dtype=paddle.float32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape((-1, 1024))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.reshape((-1, self.k, self.k)) + self.iden
        return x
```

An affine transformation matrix is predicted by a mini-network (T-net) and directly apply this transformation to the coordinates of input points. `iden` is used to make the output matrix be initialized as an identity.


### Loss Function

```python
class CrossEntropyMatrixRegularization(nn.Layer):
    def __init__(self, mat_diff_loss_scale=1e-3):
        super(CrossEntropyMatrixRegularization, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):
        loss = F.cross_entropy(pred, target)

        if trans_feat is None:
            mat_diff_loss = 0
        else:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def feature_transform_reguliarzer(trans):
    d = trans.shape[1]
    I = paddle.eye(d)
    loss = paddle.mean(
        paddle.norm(
            paddle.bmm(trans, paddle.transpose(trans, (0, 2, 1))) - I, axis=(1, 2)
        )
    )
    return loss
```

A regularization loss (with weight 0.001) is added to the softmax classification loss to make the matrix close to orthogonal. The regularization loss is necessary for the higher dimension transform to work. By combining both transformations and the regularization term, we can achieve the best performance.


### Train & Test Parameters Description:
| Name           | Type  | Default                       | Description                         |
| -------------- | ----- | ----------------------------- | ----------------------------------- |
| data_dir       | str   | "modelnet40_normal_resampled" | train & test data dir               |
| num_point      | int   | 1024                          | sample number of points             |
| batch_size     | int   | 32                            | batch size in training              |
| num_category   | int   | 40                            | ModelNet10/40                       |
| learning_rate  | float | 1e-3                          | learning rate in training           |
| max_epochs     | int   | 200                           | max epochs in training              |
| num_workers    | int   | 32                            | number of workers in dataloader     |
| log_batch_num  | int   | 50                            | log info per log_batch_num          |
| model_path     | str   | "pointnet.pdparams"           | save/load model in training/testing |
| lr_decay_step  | int   | 20                            | step_size in StepDecay              |
| lr_decay_gamma | float | 0.7                           | gamma in StepDecay                  |

## 7. Model Information

For other information about the model, please refer to the following table:
| Information       | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| Author            | Yunchong Gan                                                          |
| Date              | 2021.8                                                                |
| Framework version | Paddle 2.1.2                                                          |
| Support hardware  | GPU/CPU                                                               |
| Download link     | [pointnet.pdparams](./pointnet.pdparams)                              |
| Online operation  | [Notebook](https://aistudio.baidu.com/aistudio/projectdetail/2272519) |
