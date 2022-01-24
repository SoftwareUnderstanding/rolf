# inception-v3
This repository hosts the contributor source files for the inception-v3 model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | 001bb1c9-bbaf-48ca-bf4a-505faca870dd | 
| application_area | ImageNet | 
| task | Classification | 
| task_extended | ImageNet classification | 
| data_type | Image/Photo | 
| data_source | http://www.image-net.org/challenges/LSVRC/2012/ | 
## publication
| | |
|-|-|
| title | Rethinking the Inception Architecture for Computer Vision | 
| source | Arxiv | 
| url | https://arxiv.org/abs/1512.00567 | 
| year | 2015 | 
| authors | Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna | 
| abstract | Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=1692140599533045894&as_sdt=5 | 
| bibtex | @article{DBLP:journals/corr/SzegedyVISW15, author = {Christian Szegedy and Vincent Vanhoucke and Sergey Ioffe and Jonathon Shlens and Zbigniew Wojna}, title = {Rethinking the Inception Architecture for Computer Vision}, journal = {CoRR}, volume = {abs/1512.00567}, year = {2015}, url = {http://arxiv.org/abs/1512.00567}, archivePrefix = {arXiv}, eprint = {1512.00567}, timestamp = {Mon, 13 Aug 2018 16:49:07 +0200}, biburl    = {https://dblp.org/rec/bib/journals/corr/SzegedyVISW15}, bibsource = {dblp computer science bibliography, https://dblp.org}} | 
## model
| | |
|-|-|
| description | Inception-v3 introduces a few upgrades over the previous inception networks. It reduces representational bottlenecks as well as utilize smart factorization methods making convolutions computationally efficient. | 
| provenance | https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py | 
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .h5 | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

