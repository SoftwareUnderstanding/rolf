# GHM_paddle

## 项目说明：
### 本项目是针对 Gradient Harmonized Single-stage Detector 论文的复现工作，使用的框架是百度飞桨PaddlePaddle平台.目前完成的工作包括：
#### （1）模型组网的搭建
#### （2）模型损失函数的计算实现
#### （3）resnet50的模型对齐


## 一、论文讲解
#### 论文链接 https://arxiv.org/abs/1811.05181
#### 论文代码 https://github.com/libuyu/GHM_Detection

<img src="https://github.com/tsdlrh/Blog_image/blob/master/1.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/2.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/3.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/4.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/5.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/6.JPG" width="800px">
<img src="https://github.com/tsdlrh/Blog_image/blob/master/7.JPG" width="800px">


#### 简介：

单阶段检测在训练时存在正负样本的差距，针对easy和hard样本之间的不同，从梯度的方向考虑解决这个两个问题。提出了梯度协调机制GHM,将GHM的思想嵌入到分类的交叉熵损失和用于回归的Smooth-L1损失中，在COCO数据集上，mAP达到了41.6的优良效果。


#### 背景：

单阶段检测器在训练时面临的最大问题就是easy和hard样本，以及positive和negative样本之间的不平衡，easy样本的数量多以及背景样本的出现，影响了检测器的训练，这种问题在二阶段检测中并不存在。相关的研究技术包括，OHEM样本挖掘技术和Focal Loss函数。其中OHEM技术丢弃了大部分样本，训练也不是很高效，Focal Loss函数引入了两个超参数，需要进行大量的实验进行调试，同时Focal Loss是一种静态损失，对数据集的分布并不敏感。本论文中指出，类别不平衡问题主要归结于难度不平衡问题，而难度不平衡问题可以归结于正则化梯度分布（gradient norm）的不平衡,如果一个正样本很容量被分类，则模型从该样本中得到的信息量较少，或者说产生的梯度信息很少。从整体上看，负样本多为easy样本，正样本多为hard样本。因此，两种不平衡可以归结于属性上的不平衡。论文的主要贡献：（1）揭示了单阶段检测器在gradient norm分布方面存在显著不足的基本原理，并且提出了一种新的梯度平衡机制(GHM)来处理这个问题。（2）提出了GHM-C以及GHM-R，它们纠正了不同样本的梯度分布，并且对超参数具有鲁棒性。（3）通过使用GHM，我们可以轻松地训练单阶段检测器，无需任何数据采样策略，并且在COCO基准测试中取得了state-of-the-art的结果。


#### GHM思想：


#### (1) GHM-C Loss
对于一个候选框，它的真实便签为p*∈{0,1}，预测的值为p∈[0,1],采用二元交叉熵损失函数,那么梯度的模值定义为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/10.JPG" width="400px">
其中g代表了这个样本的难易程度以及它对整个梯度的贡献。

训练样本的梯度密度函数为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/1.JPG" width="400px">
其中gk为第k个样本的gradient norm.

g的gradient norm为在以g为中心，长度为ε的区域内的样本数，并且由该区域的有效长度进行归一化。定义梯度密度参数
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/3.JPG" width="200px">
N为样本总数

根据梯度密度参数，可以得到分类问题的损失平衡函数：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/4.JPG" width="400px">

#### (2) GHM-R Loss
Smooth L1损失函数为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/5.JPG" width="400px">

Smooth L1关于ti的导数为：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/6.JPG" width="400px">

对于所有|d|>δ的样本都具有gradient norm,这就不可能仅仅依靠gradient norm来区分不同属性的样本，为了在回归Loss上应用GHM,将传统的SL1损失函数，改变为ASL1形式
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/7.JPG" width="400px">

当d很小时，近似为一个方差函数L2 Loss,当d很大时，近似为一个线性损失L1 Loss，具有较好的平滑性，其偏导存在且连续，将GHM应用于回归Loss的结果如下：
<img src="https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/9.JPG" width="400px">


## 二、论文复现
### （1）模型组网的搭建

```
GHM_detection
|	├── backbone:Resnet50
|	├── necks:FPN
|	├── bbox_head:Retina_head
|	├── data
|	│   ├── coco
|	│   │   ├── annotations
|	│   │   ├── train2017
|	│   │   ├── val2017
|	│   │   ├── test2017
```


```
项目文件功能说明

-anchors.py 实现anchor_head代码

-utils.py 实现BasicBlock,Bottleneck,BBoxTransform和ClipBox函数功能

-retina_net.py 模型组网，搭建了resnet50+fpn+retina_head的模型结构

-losses.py 实现Focal_loss函数

-ghm_loss.py 实现GHM_C loss和GHM_R loss函数

-dataloader.py 实现COCO数据的加载

```

### ResNet50的paddle实现核心代码：
```python
#Resnet50的搭建
class ResNet(nn.Layer):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()#此处采用FOCAloss

        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

```


### FPN的paddle实现核心代码：

```Python
class PyramidFeatures(nn.Layer):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2D(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2D(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2D(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2D(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]
 ```       
        

### 分类回归模型的paddle实现核心代码：
```Python
class RegressionModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2D(feature_size, num_anchors * 4, kernel_size=3, padding=1)

```

```Python
class ClassificationModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2D(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

```        
       

### （2）模型损失函数的计算实现

### ghm_loss的paddle实现

```Python
class GHMC(nn.Layer):
    """
    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """    
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if paddle.to_tensor(pred).dim() != paddle.to_tensor(target).dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target, label_weight
        edges = self.edges
        mmt = self.momentum
        weights = paddle.zeros_like(pred)

        # gradient length
        g = paddle.abs(paddle.nn.functional.sigmoid(pred).detach() - target)

        valid = label_weight > 0
        tot = max(valid.sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]).logical_and((g < edges[i + 1]).logical_and(paddle.to_tensor(valid)))
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights = tot / self.acc_sum[i]
                else:
                    weights = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, paddle.to_tensor(target), paddle.to_tensor(weights), reduction='sum') / tot
        return loss * self.loss_weight

#GHM_R Loss损失函数，将GHM思想用于回归的Smooth L1损失函数
class GHMR(nn.Layer):
    """
    Args:
        mu (float): The parameter for the Authentic Smooth L1 loss.
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        loss_weight (float): The weight of the total GHM-R loss.
    """    
    def __init__(
            self,
            mu=0.02,
            bins=10,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None):
        """   
        Args:
            pred (float tensor of size [batch_num, 4 (* class_num)]):
                The prediction of box regression layer. Channel number can be 4
                or 4 * class_num depending on whether it is class-agnostic.
            target (float tensor of size [batch_num, 4 (* class_num)]):
                The target regression values with the same size of pred.
            label_weight (float tensor of size [batch_num, 4 (* class_num)]):
                The weight of each sample, 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = paddle.sqrt(paddle.to_tensor(diff * diff + mu * mu)) - mu

        # gradient length
        g = paddle.abs(paddle.to_tensor(diff) / paddle.sqrt(paddle.to_tensor(mu * mu + diff * diff))).detach()
        weights = paddle.zeros_like(g)

        valid = label_weight > 0

        tot = max(label_weight.sum().item(), 1.0)

        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = ((g >= edges[i]).logical_and((g < edges[i + 1]).logical_and(paddle.to_tensor(valid)))).astype(int)

            num_in_bin = inds.sum().item()

            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    weights = tot / self.acc_sum[i]

                else:
                    weights = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight

```

### （3）resnet50的模型对齐
![image](https://github.com/tsdlrh/Blog_image/blob/master/%E7%AE%A1%E7%90%86%E5%91%98%E7%99%BB%E5%BD%95/11.JPG)


### 其他

### 由于时间仓促，目前只完成了对于网络模型的组网搭建部分，对GHM_Loss的函数计算进行了Paddle实现，并对resnet50进行了前向对齐，编写了数据加载dataloader.py
### 接下来的任务还需要对necks和bbox_head进行对齐,对模型进行训练和调试以及反向对齐和loss对齐，验证论文的效果.工作后续更新

