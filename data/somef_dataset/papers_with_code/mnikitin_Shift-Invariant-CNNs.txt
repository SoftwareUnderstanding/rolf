# Shift-Invariant-CNNs
Gluon implementation of anti-aliased CNNs: https://arxiv.org/abs/1904.11486<br/>
Based on original PyTorch implementation: https://github.com/adobe/antialiased-cnns

## CIFAR-10 experiments

### Usage
Example of training *resnet20_v1* with *anti-aliasing* and *random crop* augmentation:<br/>
```
python3 train_cifar10.py --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --num-epochs 186 --lr 0.003 --lr-decay 0.1 --lr-decay-epoch 81,122 --wd 0.0001 --optimizer adam --model cifar_resnet20_v1 --antialiasing --random-crop
```

### Results
<table>
  <tr>
    <th>Model</th>
    <th>random crop</th>
    <th>anti-aliasing</th>
    <th>Train accuracy</th>
    <th>Test accuracy<br></th>
  </tr>
  <tr>
    <td rowspan="4">cifar_resnet20_v1</td>
    <td align="center">✘</td>
    <td align="center">✘</td>
    <td align="center">1.0000</td>
    <td align="center">0.8879</td>
  </tr>
  <tr>
    <td align="center">✘</td>
    <td align="center">✔</td>
    <td align="center">1.0000</td>
    <td align="center">0.9026</td>
  </tr>
  <tr>
    <td align="center">✔</td>
    <td align="center">✘</td>
    <td align="center">0.9918</td>
    <td align="center">0.9165</td>
  </tr>
  <tr>
    <td align="center">✔</td>
    <td align="center">✔</td>
    <td align="center">0.9960</td>
    <td align="center"><b>0.9184</b></td>
  </tr>
  <tr>
    <td rowspan="4">cifar_resnet20_v2</td>
    <td align="center">✘</td>
    <td align="center">✘</td>
    <td align="center">1.0000</td>
    <td align="center">0.8850</td>
  </tr>
  <tr>
    <td align="center">✘</td>
    <td align="center">✔</td>
    <td align="center">0.9999</td>
    <td align="center">0.9051</td>
  </tr>
  <tr>
    <td align="center">✔</td>
    <td align="center">✘</td>
    <td align="center">0.9891</td>
    <td align="center"><b>0.9114</b></td>
  </tr>
  <tr>
    <td align="center">✔</td>
    <td align="center">✔</td>
    <td align="center">0.9953</td>
    <td align="center">0.9084</td>
  </tr>
</table>
