# Pytorch Implementation of UNet family based on Detectron2

Currently supports:

UNet:  https://arxiv.org/abs/1505.04597

UNet++: https://arxiv.org/abs/1807.10165

ResUNet: https://arxiv.org/pdf/1711.10684.pdf (Not checked if the implementation is correct)

More will be added in the future...

This implementation fits business scenarios, in which you may want to reduce the depths/width of the network or add/reduce down-sampling stages. This implementation is more flexible, you can easily modify these in configs.

***All our works are in projects/UNet***

## NOTES:

1. All those implementations have ***NOT*** been tested on official benchmarks. However, these were tested on my business datasets doing medical segmentation and achieved convincing performance. 
2. A lot of details in implementation are unclear in papers, in these cases, I usually use other implementations' solution. (For example, whether convolutional layers in UNet uses padding=0 or padding=1, since the size of feature map reduced, as shown in paper, I believe there is no padding in convolutional layers. However, I did pad the feature maps, since other implementations did).

## Get Started

1. You have to correctly setup detectron2, see https://github.com/facebookresearch/detectron2. Here is the installation instructuion: https://detectron2.readthedocs.io/tutorials/install.html. You need to have Linux or macOS with Python >= 3.6, Pytorch >= 1.6.

2. Go to detectron2/projects/UNet/configs, choose the a config file you wish to use, currently the configs allow you to modify:

   - Channels of hidden layers: official implementation is 64-128-256-512-1024, you can change it to 16-32-32-64 or whatever you want, change the UNET_CHANNELS entry in configs.
   - INPUT.MIN_SIZE_TRAIN is used as resizing your image, if you want your input image to be resized to (512, 512), just put (512, 512) in that entry.
   - CROP.SIZE is used in random cropping.
   - NUM_CLASSES, if you choose NUM_CLASSES=3, you have to prepare your sem_seg_masks with pixel values 0, 1, 2 representing three classes. 
   - In the config of UNet++, there is an entry called UNETPP_USE_SUBNET_IN_INFERENCE, that means the entire net is used for training, but only some subnets are used in inference, for more details, see UNet++ paper, leave it as 0 if you do not want this feature.

3. Prepare your own dataset, put your dataset under data, you have to **Write Your Own DataLoading Function**. All you have to do is modify load_train_data() and load_val_data() function in data/register_dataset.py, that function does arbitrary things and returns a list[dict], where each dict represents an image and should contain following keys:

   - file_name (str): the full path to your image

   - height (int): the height of input image

   - width (int): the width of input image

   - image_id (str or int): a unique id appointed to a specific image

   - sem_seg_file_name (str): the full path to your semantic segmentation masks, where for each mask,

   ​                        you need to use a pixel value for a class, starting from 0, 1, ...

   Briefly, your function will be like:

   ```python
   def load_train_data():
       dataset = []
   	for image in your_dataset:
       	img_dict = {
               "file_name": path/to/image,
               "height": height,
               "width": width,
               "image_id": a_unique_image_id,
               "sem_seg_file": path/to/sem_seg_file,
           }
           dataset.append(img_dict)
       return dataset
   ```

4. After all these preparations, run (Assume you want to run UNet) :

   ```console
   cd projects/UNet
   python train_net.py --config-file configs/Base-UNet-DS16-Semantic.yaml
   ```

5. Model after training will be saved to /output/, modify SOLVER details and training details in config, example usage:

   ```console
   python train_net.py --config-file configs/Base-UNet-DS16-Semantic.yaml \ 
   MODEL.WEIGHTS path/to/your/model \
   SOLVER.BASE_LR 0.001 \
   SOLVER.MAX_ITER 50000 
   ```

   Check detectron2/detectron2/config/defaults.py for all config entries that you can modify

