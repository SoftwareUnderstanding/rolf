# Cascade-point-rcnn
a muilti-stage- 3d detector based on PointRCNN (https://github.com/sshaoshuai/PointRCNN) 
and Cascade-rcnn https://arxiv.org/abs/1712.00726

sh build_and_install.sh
## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```
 Generate the augmented offline scenes by running the following command:
```
python generate_aug_scene.py --class_name Car --split train --aug_times 4
```
(a) Train RCNN network with fixed RPN network to use online GT augmentation: Use `--rpn_ckpt` to specify the path of a well-trained RPN model and run the command as follows:
```
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ./PointRCNN.pth --ouput_dir ./cascade_output
```
To evaluate a single checkpoint, run the following command with `--ckpt` to specify the checkpoint to be evaluated:
```
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ./cascade_output/ckpt/checkpoint_epoch_?.pth --batch_size 4 --eval_mode rcnn 
