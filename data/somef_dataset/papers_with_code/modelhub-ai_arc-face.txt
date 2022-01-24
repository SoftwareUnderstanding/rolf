# arc-face
This repository hosts the contributor source files for the arc-face model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).
## meta
| | |
|-|-|
| id | 02f2f15c-3285-44a4-8119-b298623b7acf | 
| application_area | Computer Vision | 
| task | Recognition | 
| task_extended | Facial Detection & Recognition | 
| data_type | Image/Photo | 
| data_source | https://www.msceleb.org/ | 
## publication
| | |
|-|-|
| title | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | 
| source | arxiv | 
| url | https://arxiv.org/abs/1801.07698 | 
| year | 2018 | 
| authors | Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou | 
| abstract | One of the main challenges in feature learning using Deep Convolutional Neural Networks (DCNNs) for large-scale face recognition is the design of appropriate loss functions that enhance discriminative power. Centre loss penalises the distance between the deep features and their corresponding class centres in the Euclidean space to achieve intra-class compactness. SphereFace assumes that the linear transformation matrix in the last fully connected layer can be used as a representation of the class centres in an angular space and penalises the angles between the deep features and their corresponding weights in a multiplicative way. Recently, a popular line of research is to incorporate margins in well-established loss functions in order to maximise face class separability. In this paper, we propose an Additive Angular Margin Loss (ArcFace) to obtain highly discriminative features for face recognition. The proposed ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere. We present arguably the most extensive experimental evaluation of all the recent state-of-the-art face recognition methods on over 10 face recognition benchmarks including a new large-scale image database with trillion level of pairs and a large-scale video dataset. We show that ArcFace consistently outperforms the state-of-the-art and can be easily implemented with negligible computational overhead. We release all refined training data, training codes, pre-trained models and training logs, which will help reproduce the results in this paper. | 
| google_scholar | https://scholar.google.com/scholar?oi=bibs&hl=en&cites=13816119281473749224 | 
| bibtex | @article{DBLP:journals/corr/abs-1801-07698, author= {Jiankang Deng and Jia Guo and Stefanos Zafeiriou}, title = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition}, journal = {CoRR}, volume = {abs/1801.07698}, year = {2018}, url = {http://arxiv.org/abs/1801.07698}, archivePrefix = {arXiv}, eprint = {1801.07698}, timestamp = {Mon, 13 Aug 2018 16:46:52 +0200}, biburl = {https://dblp.org/rec/bib/journals/corr/abs-1801-07698}, bibsource = {dblp computer science bibliography, https://dblp.org}} | 
## model
| | |
|-|-|
| description | ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images. To enhance the discriminative power of softmax loss, a novel supervisor signal called additive angular margin (ArcFace) is used here as an additive term in the softmax loss. | 
| provenance | https://github.com/onnx/models/tree/master/models/face_recognition/ArcFace |
| architecture | Convolutional Neural Network (CNN) | 
| learning_type | Supervised learning | 
| format | .onnx | 
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) | 
| license | model license can be viewed [here](contrib_src/license/model) | 
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).

