# Learning to Train with Synthetic Humans

This repository provides the training and evaluation code of [Learning to Train with Synthetic Humans](https://arxiv.org/abs/1908.00967). It contains a tensorflow implementation of [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf) for the [MPII multi-person pose estimation benchmark](http://human-pose.mpi-inf.mpg.de/#overview). Besides that, it contains the code for the teacher network, as described in [Learning to Train with Synthetic Humans](https://arxiv.org/abs/1908.00967).
To use the code you will need to download multiple files from the [project website](https://ltsh.is.tue.mpg.de/). All necessary files can be found in the download section in the subsection "Files for LTSH training code". Note that accessing these files requires to sign up and to agree to our license.


## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or
use the model, data and software, (the "Model & Software"), textures, software, scripts. By downloading and/or using the Model & Software (including
downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these
terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions,
you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically
terminate your rights under this [License](./LICENSE)

## Installation

a) We highly recommend the usage of virtualenv. Build a pyhthon 3 virtualenv and activate it.

b) Install dependencies ```pip install tensorflow-gpu==1.5.0, opencv-python, protobuf```. ```cd LearningToTrainWithSyntheticHumans```. ```pip install  -r requirements.txt```. ```git clone https://github.com/pdollar/coco.git``` ```cd coco/PythonAPI``` ```python setup.py instal```.

c) Install swig and build the code for post processing. ```sudo apt install swig```. ```cd ./tf_pose/pafprocess_mpi```. ```swig -python -c++ pafprocess_mpi.i && python3 setup.py build_ext --inplace```.


## Training

a) Download the [MPII pose estimation dataset](http://human-pose.mpi-inf.mpg.de/) and download [masks](https://ltsh.is.tue.mpg.de/downloads). Create a root directory ```mkdir MPII_images``` and copy the images to ```/MPII_images/images``` and the masks to ```/MPII_images/masks```.

b) Download either the [purely synthetic dataset](https://ltsh.is.tue.mpg.de/), the [mixed dataset](https://ltsh.is.tue.mpg.de/) or the [stylized dataset](https://ltsh.is.tue.mpg.de/). Download the [JsonFiles](https://ltsh.is.tue.mpg.de/), containing the labels and extract the respective the folders to ```/LearningToTrainWithSyntheticHumans/```.

c) Open ```/misc/replace_substring_infile.py``` and set "basepath" to the path of the _json folder and "pathToSynthetic" to the path of the downloaded synthetic training data. ```cd misc``` and run replace_substring_infile.py.

d) Download [models_training.7z](https://ltsh.is.tue.mpg.de/) and extract the content to the repositories root directory.
### With Teacher

With synthetic data:  
```python tf_pose/train_with_adversarial_teacher.py  --identifier=synthetic_teacher --param-idx=0```

With mixed data:  
```python tf_pose/train_with_adversarial_teacher.py --synth_data_path=./mixedData/ --identifier=mixed_teacher --param-idx=0 --mixed_data=True```

With stylized data:  
```python tf_pose/train_with_adversarial_teacher.py --synth_data_path=./stylized/ --identifier=style_teacher --param-idx=0  --mixed_data=True --stylized=True```

### Without Teacher

```python tf_pose/train_withoutTeacher.py --synth_data_path=./mixedData/ --identifier=mixed_boTeacher --param-idx=0 --gpus=1 --mixed_data=True```

### Masked condition

To mask out the loss of synthetic humans use --volumetricMask=True.

## Demo
a) Run ```python tf_pose/eval_MPI.py``` with all necessary flags.

## Citation

If you find this Software or Dataset useful in your research we would kindly ask you to cite:

```
@inproceedings{Hoffmann:GCPR:2019,
  title = {Learning to Train with Synthetic Humans},
  author = {Hoffmann, David T. and Tzionas, Dimitrios and Black, Michael J. and Tang, Siyu},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  month = sep,
  year = {2019},
  url = {https://ltsh.is.tue.mpg.de},
  month_numeric = {9}
}
```

## Support
We only support the default settings.

## Acknowledgement

We want to thank Ildoo Kim for his repository [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), as this repository is built on top of it.

## Contact
For questions, please contact [david.hoffmann@tuebingen.mpg.de](mailto:david.hoffmann@tuebingen.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
