# network-in-network
This repository hosts the contributor source files for the network-in-network model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | b73d1ee2-c1c2-4a7f-951e-6ec569a8dc98 | 
| application_area | ImageNet | 
| task | Classification | 
| task_extended | ImageNet classification | 
| data_type | Image/Photo | 
| data_source | http://www.image-net.org/ | 
## publication
| | |
|-|-|
| title | Network In Network | 
| source | arXiv | 
| url | https://arxiv.org/abs/1312.4400 | 
| year | 2014 | 
| authors | Min Lin, Qiang Chen, Shuicheng Yan | 
| abstract | We propose a novel deep network structure called 'Network In Network' (NIN) to enhance model discriminability for local patches within the receptive field. The conventional convolutional layer uses linear filters followed by a nonlinear activation function to scan the input. Instead, we build micro neural networks with more complex structures to abstract the data within the receptive field. We instantiate the micro neural network with a multilayer perceptron, which is a potent function approximator. The feature maps are obtained by sliding the micro networks over the input in a similar manner as CNN; they are then fed into the next layer. Deep NIN can be implemented by stacking mutiple of the above described structure. With enhanced local modeling via the micro network, we are able to utilize global average pooling over feature maps in the classification layer, which is easier to interpret and less prone to overfitting than traditional fully connected layers. We demonstrated the state-of-the-art classification performances with NIN on CIFAR-10 and CIFAR-100, and reasonable performances on SVHN and MNIST datasets. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=3211704355758672916&as_sdt=5 | 
| bibtex | @article{DBLP:journals/corr/LinCY13, author = {Min Lin and Qiang Chen and Shuicheng Yan}, title = {Network In Network}, journal = {CoRR}, volume = {abs/1312.4400}, year = {2013}, url = {http://arxiv.org/abs/1312.4400}, archivePrefix = {arXiv}, eprint = {1312.4400}, timestamp = {Mon, 13 Aug 2018 16:47:07 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/LinCY13}, bibsource = {dblp computer science bibliography, https://dblp.org}} | 
## model
| | |
|-|-|
| description | This network consists of multi-layer perceptron convolutional layers which use multilayer perceptrons to convolve the input and a global average pooling layer as a replacement for the fully connected layers in conventional CNN. | 
| provenance | https://mxnet.apache.org/model_zoo/index.html | 
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .json | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

