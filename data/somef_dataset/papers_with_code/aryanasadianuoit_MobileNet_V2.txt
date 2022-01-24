# MobileNet_V2
Unofficial implementation of MobileNet-V2 in PyTorch.

Reference : <a href="https://arxiv.org/pdf/1801.04381.pdf">https://arxiv.org/pdf/1801.04381.pdf</a>
<br>
<section>

MobileNet-V2 addresses the challenges of using deep learning models in resource constraints environments, e.g., mobile devices and embedded systems. The main idea behind MobileNet-V2 is to replace most of the regular convolutional layers in a conventional CNN model with <b>Inverted Residual blocks</b>. These blocks are made of <b>depth-wise</b> convolutions (with the kernel size of 3 * 3), <b>point-wise</b> convolutions (with the kernel size of 1 * 1), both equipped with <b>non-linear activations</b>, and a final <b>point-wise</b> convolution with linear mapping. The figure below depicts the mechanism of depth-wise, and point-wise convolutional layers, as well as inverted residual blocks.

<br>
<p><b>PS-1</b>: In the original paper, the activation function for non-linear transformation is <b>ReLU-6</b>. In this implementation, I have replaced ReLU-6 with the regular ReLU.</p>
<p><b>PS-2</b>: In the original paper, the last layer is a regular convolutional layer with <b>1000 channels</b> (number of ImageNet classes), and <b>(1,1)</b> spatial size. In this implementation, <b>a combination of dropout(0.2), and a linear layer</b> has been used instead of the mentioned CNN layer (inspired by the <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py">PyTorch</a> implementation)</p>
</section>
<br>
<section>
  <h2>Architecture</h2>
  <p>The following table is the architecture of MobileNet-V2. </p>
  <br>
  <table>
  <tr>
    <th align="center">Input size</th>
    <th align="center">layer/module</th> 
    <th align="center">expansion rate (t)</th>
    <th align="center"># out-channels (c)</th>
    <th align="center"># of layer/module (n)</th>
    <th align="center">stride (s)</th>
  </tr>
  <tr>
    <td align="center">(3,224,224)</td>
    <td align="center">Conv2d</td> 
    <td align="center">-</td>
    <td align="center">32</td>
    <td align="center">1</td>
    <td align="center">2</td>
  </tr>
   <tr>
    <td align="center">(32,112,112)</td>
    <td align="center">bottleneck</td> 
    <td align="center">1</td>
    <td align="center">16</td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>
  <tr>
    <td align="center">(16,112,112)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">24</td>
    <td align="center">2</td>
    <td align="center">2</td>
  </tr>
   <tr>
    <td align="center">(24,56,56)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">32</td>
    <td align="center">3</td>
    <td align="center">2</td>
  </tr>
   <tr>
    <td align="center">(32,28,28)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">64</td>
    <td align="center">4</td>
    <td align="center">2</td>
  </tr>
   <tr>
    <td align="center">(64,14,14)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">96</td>
    <td align="center">3</td>
    <td align="center">1</td>
  </tr>
   <tr>
    <td align="center">(96,14,14)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">160</td>
    <td align="center">3</td>
    <td align="center">2</td>
  </tr>
   <tr>
    <td align="center">(160,7,7)</td>
    <td align="center">bottleneck</td> 
    <td align="center">6</td>
    <td align="center">320</td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>
   <tr>
    <td align="center">(320,7,7)</td>
    <td align="center">Conv2d (1x1)</td> 
    <td align="center">-</td>
    <td align="center">1280</td>
    <td align="center">1</td>
    <td align="center">1</td>
  </tr>
   <tr>
    <td align="center">(1280,7,7)</td>
    <td align="center">avgpool (7x7)</td> 
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">1</td>
    <td align="center">-</td>
  </tr>
   <tr>
    <td align="center">(1280,1,1)</td>
    <td align="center">Conv2d (1x1)</td> 
    <td align="center">-</td>
    <td align="center">num_classes (1000)</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>

</table>
  </section>
