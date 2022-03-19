# Face image deblurring: A Two Phases Data-Driven Learning Strategy
- This project was motivated by [CycleGAN](https://github.com/vanhuyz/CycleGAN-TensorFlow)and [Scale-recurrent Network ](https://github.com/jiangsutx/SRN-Deblur).
- CycleGAN Original paper: https://arxiv.org/abs/1703.10593
- Scale-recurrent Network Original paper: http://www.xtao.website/projects/srndeblur/srndeblur_cvpr18.pdf

### Our main contributions in this work are:
-Rather than using CycleGAN [6] just for data augmentation as it is communally used in some recent works. Instead we present a use case where we can exploit this algorithm for data labeling moreover than just data augmentation.
-To better address the face image blurring problematic, we present a sequential learning strategy in a learning chain consists of an unsupervised learning based-algorithm in charge of data labeling and a supervised learning-guided algorithm taking charge of face image recovery.
-We investigated the face image deblurring impact on the face detection accuracy.

### The first training phase: CycleGAN  
<img src="./imgs/phase1.PNG" width="100%" alt="Real Photo">

### The second training phase: SNR using the trained CycleGAN as a backbone. 
<img src="./imgs/phase2.PNG" width="100%" alt="Testing Dataset">

### Visual comparisons on our testing dataset. from left to right: Ground truth, Blurred input, Tao et al, Yuan et al, Zhu et al, Ours.
<img src="./imgs/results1.PNG" width="100%" alt="More Cases">

### Visual comparisons on real blurred face images. From left to right: blurred input, Tao et al, Yuan et al , Zhu et al , Ours. 
<img src="./imgs/results2.PNG" width="100%" alt="More Cases">

### FaceBox, Face detection algorithm performance on our face deblurring results: a, b and c respectively show the ground truth, blurred and restored version.
<img src="./imgs/face.PNG" width="100%" alt="More Cases">


# CycleGAN
## Requirements
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://pypi.org/project/Keras/)
## Dataset
Samples of dataset we use are in the **CycleGAN/CycleGAN_Data** folder,for better performance collect more real face unlabeled clear/blurred image training data .

* Write the dataset to tfrecords
```bash
$ cd CycleGAN/CycleGAN_Code
$ python build_data.py --X_input_dir  CycleGAN/CycleGAN_dataset/trainA \
	               --Y_input_dir CycleGAN/CycleGAN_dataset/trainB \
	--X_output_file CycleGAN//CycleGAN_dataset/blurred.tfrecords \
        --Y_output_file CycleGAN//CycleGAN_dataset/sharp.tfrecords
```

## Training

```bash
$ cd CycleGAN/CycleGAN_Code
$ python train.py --X CycleGAN/CycleGAN_dataset/blurred.tfrecords \
		   --Y CycleGAN/CycleGAN_dataset/sharp.tfrecords \
				   --skip False
```

To change other default settings, you can check [train.py](https://github.com/QLightman/VRAR-Course-Project/blob/master/%20CycleGAN_Code/train.py)


## Check TensorBoard to see training progress and generated images.
```
$ tensorboard --logdir checkpoints/${datetime}
```

## Export model
You can export from a checkpoint to a standalone GraphDef file as follow:

```bash
$ python export_graph.py --checkpoint_dir checkpoints/${datetime} \
                          --XtoY_model blurred2sharp.pb \
                          --YtoX_model sharp2blurred.pb \
                          --image_size 256
```

## Inference
After exporting model, you can use it for inference. For example:
```bash
cd /CycleGAN
python inference.py --model CycleGAN_Model/sharp2blurred.pb \
                     --input input_sample.jpg \
                     --output output_sample.jpg \
                     --image_size 256
```

## Pretrained Models
Our pretrained models are in the **CycleGAN_Model** folder. 

# Scale-recurrent Network 
## Prerequisites
- Python2.7
- Scipy
- Scikit-image
- numpy
- Tensorflow 1.4 with NVIDIA GPU or CPU (cpu testing is very slow)
## Installation
Clone this project to your machine. 

```bash
git clone https://github.com/jiangsutx/SRN-Deblur.git
cd SRN-Deblur
```
## Training
Using the trained CycleGAN We inferred a blurred version of [CelebA dataset](https://www.kaggle.com/jessicali9530/celeba-dataset).CebebA is a benchmarked clear face image dataset downloadable from this link: https://www.kaggle.com/jessicali9530/celeba-dataset 
In order to build the a clear/blurred labed face image dataset:Run the trained CycleGAN inference on whole CelebA dataset using the below command on line.
```bash
cd /CycleGAN
python inference.py --model CycleGAN_Model/sharp2blurred.pb \
                     --input input_sample.jpg \
                     --output output_sample.jpg \
                     --image_size 256
```
Please put the dataset into `training_set/`. And the provided `datalist.txt` can be used to train the model, follow the template and adapt its contain to your data order and location.  

Hyper parameters such as batch size, learning rate, epoch number can be tuned through command line:

```bash
cd /SRN
python run_model.py --phase=train --batch=16 --lr=1e-4 --epoch=4000
```
## Testing

We provide pretrained models inside `checkpoints/`.

To test blur images in a folder, just use arguments 
`--input_path=<TEST_FOLDER>` and save the outputs to `--output_path=<OUTPUT_FOLDER>`.
For example:

```bash
python run_model.py --input_path=./testing_set --output_path=./testing_res
```

If you have a GPU, please include `--gpu` argument, and add your gpu id to your command. 
Otherwise, use `--gpu=-1` for CPU. 

```bash
python run_model.py --gpu=0
```

To test the model, pre-defined height and width of tensorflow 
placeholder should be assigned. 
Our network requires the height and width be multiples of `16`. 
When the gpu memory is enough, the height and width could be assigned to 
the maximum to accommodate all the images. 

Otherwise, the images will be downsampled by the largest scale factor to 
be fed into the placeholder. And results will be upsampled to the original size.

According to our experience, `--height=720` and `--width=1280` work well 
on a Gefore GTX 1050 TI with 4GB memory. For example, 

```bash
python run_model.py --height=720 --width=1280
```




