# Problem Statement 6

The problem statement demands a means of communication between two persons, one of whom is deaf and dumb. The process requires Sign Language which is assumed to be known by the deaf and dumb person but absolutely unknown to the other. The app should aid as a translator. Whatever the normal person says must be mapped to the sign language by the app OR whatever sign the differently abled person generates must be converted into proper audio.  

The system should be able to generate gestures for voice.  
The system should be able to generate voice for gestures.  
  
  
## Dataset
Download https://www.kaggle.com/kuzivakwashe/significant-asl-sign-language-alphabet-dataset and move each ASL class's folder (currently 28 classes) inside data/training/ (inside the repo's folder).                                                                                 
For the next dataset, run  
```
python open_images_downloader.py --root ~/data/open_images --class_names "Human hand" --num_workers 20
```  
which uses https://storage.googleapis.com/openimages/web/index.html for labeled dataset. Refer to  qfgaohao's https://github.com/qfgaohao/pytorch-ssd for more information.
  
  
## Models
For hand detection, SSD MobileNetV1 is used; and for sign language prediction, ResNet34 architecture.
```
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
```                                                 
```
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt                                                           
```
  
 
## Training
To replicate the exact results we saw in our project, run the following lines.  
### For MobileNetV1 part:
```
python train_ssd.py --dataset_type open_images --datasets ~/data/open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.001 --t_max 100 --validation_epochs 4 --num_epochs 20 --base_net_lr 0.001  --batch_size 32 --balance_data --num_workers 0
```                                                                                                             
We stopped the training at epoch 8 (due to time constraint) and got good results. Feel free to train it for longer using different settings. For faster training, check out https://github.com/qfgaohao/pytorch-ssd/issues/19#issuecomment-467299010 
### For ResNet part:
Run resnet_model.ipynb **You need fastai installed in your environment to run that.**
  
  
 ## Testing

After changing the model_path in line 26 of sign_lang_predictor.py, run 
 ```
 python sign_lang_predictor.py
 ```                                                                                                         
   


## References
https://arxiv.org/abs/1512.02325 by Wei Liu and et al.                                                                                  
  
https://arxiv.org/abs/1512.03385 by Kaiming He and et al.       
  
https://www.fast.ai/ by Jeremy Howard and et al.
