# Deeplearning_project_STL_10
본 Repo는 STL 10에 대한 학습을 위해 만들어졌다. 단순히 STL 10을 학습시키는게 아니라, train set은 class당 500개씩 총 5000개, total model parameter은 2M로 제한하여 학습시키는 것이 목표이다. 물론 외부 데이터나, 외부 trained model weight의 사용은 하지 않고, scratch 상태에서 model을 학습시키는 것이 목적이다. 정리되지 않은 실험은 [Deeplearning_project_STL_10_Old](https://github.com/kun-woo-park/Deeplearning_project_STL_10_Old)에서 확인 할 수 있다.

## Implementation and several tries
```bash
git clone https://github.com/kun-woo-park/Deeplearning_project_STL_10.git
cd Deeplearning_project_STL_10
python3 train.py
```
Train set과 validation set간의 correlation을 줄이기 위해 주어진 dataset의 split을 먼저 진행하였다. train set 4000개와 validation set 1000개로 나누어 두었다. 여러 실험을 진행 후에, ResNet구조를 기반으로 model의 구조를 완성시켰다. 구현된 model 구조는 다음과 같다.

### Custom ResNet model
```python
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=4, stride=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.LeakyReLU(0.1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 30, layers[0])
        self.layer2 = self._make_layer(block, 60, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 96, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def Model(pretrained: bool = False, progress: bool = True, **kwargs):
    
    kwargs['groups'] = 1
    kwargs['width_per_group'] = 64
    return _resnet('resnet', Bottleneck, [4, 9, 8], pretrained, progress, **kwargs)
    
```
여기에 다양한 기법을 활용하여 학습 성능을 높였다. 사용된 기법들은 data augmentation, label smoothing, learning rate scheduling, fix train-test resolution discrepancy 들이 있다.

### Data augmentation
Train set이 총 5000개로 많지 않은 숫자였기 때문에, data augmentation을 통해 총 10만개의 data로 data 양을 늘렸다. 사용된 augmentation은 다음과 같다.

#### Augmentation code for train
```python
transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.RandomResizedCrop(70),
        transforms.ColorJitter(.3,.3,.3,.3),
        transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
```
#### Augmentation code for fine tuning
```python
transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.RandomResizedCrop(120),
        transforms.ColorJitter(.3,.3,.3,.3),
        transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
```

### Label smoothing
Criterion을 label smoothing을 적용한 CrossEntropyLoss를 사용해 보았으나, 성능 향상이 거의 없어 최종 실험에서는 적용을 제외했다. 구현된 코드는 다음과 같다.
#### Code for label smoothing loss
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
```
### Learning rate scheduling
여러 learning rate sheduler을 사용해 보았지만, cosine annealing scheduler가 가장 이상적이었다. 추가적인 수정을 통해 cosine annealing(warm restart) 이 update되는 step마다 최대값이 감소하도록 다시 구현하였다. 구현된 learning rate scheduling은 아래와 같다.
#### Overview of learning rate scheduling
<img src="./img/lr_sch.png" width="50%">

### Fix train-test resolution discrepancy
이 기법은 https://arxiv.org/pdf/1906.06423.pdf 논문을 토대로 적용하였다. 단순하게 train시에 augmented 되면서 잘려 있던 이미지의 해상도를 낮추고(crop되어 augmented될 때 물체의 사이즈가 커지지 않고 유지시키기 위해) test시에는 해상도를 높여서 train시의 물체와 test시의 물체가 비슷한 해상도(비슷한 크기로)로 보이도록 하는 것이다. Data augmentation에서도 확인할 수 있듯, 이를 위해 train시에는 70으로 resize하여 300 epochs를 학습하였고, test시를 위해 120으로 resize하여 추가적으로 140 epochs을 학습하였다(fine tuning).

## Result
최종 결과에 대한 validation accuracy의 그래프는 아래와 같다. x축은 epoch 수이고, y축은 validation accuracy이다. 최종적으로 2M의 제한 이내에서 84%의 정확도까지 개선하였다.
#### Result of validation accuracy
<img src="./img/loss.png" width="80%">
