# Towards Better Exploiting Convolutional Neural Networks for Remote Sensing Scene Classification

This repository is part of the codes used in this [paper](http://www.sciencedirect.com/science/article/pii/S0031320316301509).
Specifically, this code implements the **feature extraction** of pre-trained deep-learning models using [Caffe framework](http://caffe.berkeleyvision.org/).
This code, **which is very easy to use**, was created following this [tutorial](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html).

The networks implemented in this code and that can be used to extract deep features of any dataset are:

  - [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
  - [VGG16](https://arxiv.org/abs/1409.1556)
  - [GoogleNet](https://arxiv.org/abs/1409.4842)

### Dataset

The Brazilian Coffee Scenes Dataset exploited in this work can be found [here](http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-coffee-scenes-dataset/).

This dataset is a composition of scenes taken by SPOT sensor in 2005 over four counties in the State of Minas Gerais, Brazil: Arceburgo, Guaranesia, Guaxupé and Monte Santo. It has many intraclass variances caused by different crop management techniques. Also, coffee is an evergreen culture and the South of Minas Gerais is a mountainous region, which means that this dataset includes scenes with different plant ages and/or with spectral distortions caused by shadows. 

### Features

Some of the features exploited in this work can be downloaded here:
  - [RS19 low-level features](https://www.dropbox.com/s/d19adxpn9jk5kuh/RS19_low_level_features.zip?dl=0)
  - [RS19 deep features](https://www.dropbox.com/s/e994jhhdl0hz5i3/RS19_deep_features.zip?dl=0)
  - [UCMerced low-level features](https://www.dropbox.com/s/n3vtwi80ewdq900/UCMerced_low_level_features.zip?dl=0)
  - [UCMerced deep features](https://www.dropbox.com/s/0rdvcv5fc0wkar8/UCMerced_deep_features.zip?dl=0)

### Citing

If you use this code or features in your research, please consider citing:

    @article{nogueira2017towards,
		title={Towards Better Exploiting Convolutional Neural Networks for Remote Sensing Scene Classification},
		author={Nogueira, Keiller and Penatti, Ot{\'a}vio AB and Santos, Jefersson A dos},
		journal=pr,
		volume={61},
		number={1},
		pages={539--556},
		year={2017},
		publisher={Elsevier}
	}
	
	@inproceedings{penatti2015deep,
		title={Do deep features generalize from everyday objects to remote sensing and aerial scenes domains?},
		author={Penatti, Otavio and Nogueira, Keiller and dos Santos, Jefersson},
		booktitle=cvprw,
		pages={44--51},
		year={2015}
	}
