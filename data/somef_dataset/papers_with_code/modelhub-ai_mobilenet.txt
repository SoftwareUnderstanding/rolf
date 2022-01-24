# mobilenet
This repository hosts the contributor source files for the mobilenet model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | 016c5f2a-9b58-44a9-bc93-10dc67385035 | 
| application_area | ImageNet | 
| task | Classification | 
| task_extended | ImageNet classification | 
| data_type | Image/Photo | 
| data_source | http://www.image-net.org/ | 
## publication
| | |
|-|-|
| title | MobileNetV2: Inverted Residuals and Linear Bottlenecks | 
| source | arXiv | 
| url | https://arxiv.org/abs/1801.04381 | 
| year | 2018 | 
| authors | Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen | 
| abstract | In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=5034558864053164025&as_sdt=5 | 
| bibtex | @article{DBLP:journals/corr/abs-1801-04381, author = {Mark Sandler and Andrew G. Howard and Menglong Zhu and Andrey Zhmoginov and Liang{-}Chieh Chen}, title = {Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation}, journal = {CoRR}, volume = {abs/1801.04381}, year = {2018}, url = {http://arxiv.org/abs/1801.04381}, archivePrefix = {arXiv}, eprint = {1801.04381}, timestamp = {Mon, 13 Aug 2018 16:48:30 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/abs-1801-04381}, bibsource = {dblp computer science bibliography, https://dblp.org}} | 
## model
| | |
|-|-|
| description | MobileNet utilizes an inverted residual structure. Shortcut connections are placed between thin bottleneck layers. Intermediate expansion layer make use of depthwise convolutions to filter features. MobileNet also removes non-linearities in the narrow layers as a means to maintain representational power. | 
| provenance | https://github.com/onnx/models/tree/master/models/image_classification/mobilenet | 
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .onnx | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

