# Reference
reference github: [Tutorial to Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)
# Paper Reference
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)
# Reproducibility
1. clone github
<pre><code>!git clone https://github.com/chilung/NCTU_Adv_DNN_HW4.git
%cd /content/NCTU_Adv_DNN_HW4
</code></pre>
2. download training dataset
3. produce file `./train_images.json` in root and contain the training filename list, such as:
<pre><code>'./training_hr_images/92059.png'
'./training_hr_images/15004.png'
...
</code></pre>
## Training
### Phase 1 - training srresnet
According to the [paper](https://arxiv.org/abs/1609.04802): "We employed the trained MSE-based SRResNet network as initialization for the generator when training the actual GAN to avoid undesired local optima.The SRResNet networks were trained with a learning rate of 10^-4 and 10^6 update iterations.
<pre><code>!python train_srresnet.py --help
usage: train_srresnet.py [-h] [-r ROOT] [-c CHECKPOINT]
optional arguments:
  -h, --help            show this help message and exit
  -r ROOT, --root ROOT  the path to the root directory of model checkpoint, such as ./checkpoint
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        the path to the model checkpoint where the resume training from, such as checkpoint_srresnet_8100.pth.tar</code></pre>
### Phase 2 - training srgan
According to the [paper](https://arxiv.org/abs/1609.04802): "SRGAN was trained with 10^5 update iterations at a learning rate of 10^−4 and another 10^5 iterations at a lower rate of 10^−5"
<pre><code>!python train_srgan.py --help
usage: train_srgan.py [-h] [-r ROOT] [-c CHECKPOINT] [-s SRRESNET]
optional arguments:
  -h, --help            show this help message and exit
  -r ROOT, --root ROOT  the path to the root directory of model checkpoint, such as ./checkpoint
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        the path to the model checkpoint where the resume training from, such as checkpoint_srgan_8100.pth.tar
  -s SRRESNET, --srresnet SRRESNET
                        the filepath of the trained SRResNet checkpoint used for initialization, such as checkpoint_srresnet.pth.tar</code></pre>
## Super Resolution
<pre><code>!python output_sr.py --help
usage: output_sr.py [-h] [-o OUTPUT] [-g GAN]
optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        the path to the output directory of super resolution images.
  -g GAN, --gan GAN     the full file path to the super resolution model checkpoint, such as checkpoint_srgan.pth.tar</code></pre>
