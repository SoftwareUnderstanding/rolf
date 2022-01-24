<h1>Cross-layer pooling algorithm</h1> <br/>
This repository provides a basic implementation of Cross-layer pooling algorithm (Liu et al. 2015) in C++ (using OpenCV and Caffe) and Python (using TensorFlow). The code uses pretrained ResNet-152 network for its initialization. Refer to the paper for more details on ResNet (He et al. 2015) <b>[https://arxiv.org/abs/1512.03385]</b> and <b>[https://arxiv.org/abs/1411.7466]</b> for details on the cross-pooling method. For training the network on your own dataset, <b>CrossLayerPoolingClassifier</b> class to train a linear SVM on top of features computed through cross-layer pooling strategy or use a pre-trained SVM for predictions.

<br/><b>Code Dependencies (C++): </b>
<ol>
<li>OpenCV-3.2</li>
<li>Caffe</li>
<li>Boost</li>
<li>Pretrained ResNet-152-model.caffemodel (https://github.com/KaimingHe/deep-residual-networks)</li>
</ol>

<br/><b>Code Dependencies (TensorFlow): </b>
<ol>
<li>TensorFlow (v1.10 or newer)</li>
</ol>

<br/><b>TODOs: </b>
<ol>
<li>Add support for new classifiers along with SVM</li>
<li>Add support for larger region sizes using PCA</li>
<li>Add optimization</li>
<li>Add code profiling</li>
</ol>

## Cite

```
@article{doi:10.1093/icesjms/fsx109,
author = {Siddiqui, Shoaib Ahmed and Salman, Ahmad and Malik, Muhammad Imran and Shafait, Faisal and Mian, Ajmal and Shortis, Mark R and Harvey, Euan S and Handling editor: Howard Browman},
title = {Automatic fish species classification in underwater videos: exploiting pre-trained deep neural network models to compensate for limited labelled data},
journal = {ICES Journal of Marine Science},
volume = {75},
number = {1},
pages = {374-389},
year = {2018},
doi = {10.1093/icesjms/fsx109},
URL = {http://dx.doi.org/10.1093/icesjms/fsx109},
eprint = {/oup/backfile/content_public/journal/icesjms/75/1/10.1093_icesjms_fsx109/1/fsx109.pdf}
}
```

## License:

MIT

## Issues/Feedback:

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **shoaib_ahmed.siddiqui@dfki.de**
