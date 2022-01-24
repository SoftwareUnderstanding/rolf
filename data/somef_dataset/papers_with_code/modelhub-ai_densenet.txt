# densenet
This repository hosts the contributor source files for the densenet model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | c0048222-b29f-4719-be6b-cac790251a19 | 
| application_area | ImageNet | 
| task | Classification | 
| task_extended | ImageNet classification | 
| data_type | Image/Photo | 
| data_source | http://www.image-net.org/ | 
## publication
| | |
|-|-|
| title | Densely Connected Convolutional Networks | 
| source | arXiv | 
| url | https://arxiv.org/abs/1608.06993 | 
| year | 2016 | 
| authors | Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger | 
| abstract | Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at this https URL. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=4205512852566836101 | 
| bibtex | @article{DBLP:journals/corr/HuangLW16a, author = {Gao Huang and Zhuang Liu and Kilian Q. Weinberger}, title = {Densely Connected Convolutional Networks}, journal = {CoRR}, volume = {abs/1608.06993}, year = {2016}, url = {http://arxiv.org/abs/1608.06993}, archivePrefix = {arXiv}, eprint = {1608.06993}, timestamp = {Mon, 10 Sep 2018 15:49:32 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/HuangLW16a}, bibsource = {dblp computer science bibliography, https://dblp.org}} | 
## model
| | |
|-|-|
| description | DenseNet increases the depth of convolutional networks by simplifying the connectivity pattern between layers. It exploits the full potential of the network through feature reuse. | 
| provenance | https://github.com/flyyufelix/DenseNet-Keras | 
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .h5 | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

