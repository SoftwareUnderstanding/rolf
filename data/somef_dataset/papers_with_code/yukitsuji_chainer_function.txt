# Customized chainer's functions
- **Correlational layer**  
Proposed in this paper: https://arxiv.org/pdf/1504.06852.pdf  
Reference this code: https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/correlation_layer1d.cu

- **Spatial Dropout layer**  
Proposed in this paper: https://arxiv.org/pdf/1411.4280.pdf  
Used in ENet repository: https://github.com/yukitsuji/ENet_chainer/blob/fbbb68c77d073f2cd00cfb7bf000fdfec54e18c9/enet/models/enet_paper.py

```
from correlational_layer import correlational_layer
shape = [(1, 32, 400, 1200), (1, 32, 200, 600), (1, 32, 100, 300), (1, 32, 50, 150)][::-1]
left_img = Variable(cp.ones(shape[j], dtype='f'))
right_img = Variable(cp.ones(shape[j], dtype='f'))
out = correlational_layer(left_img, right_img, max_displacement=40)
```
