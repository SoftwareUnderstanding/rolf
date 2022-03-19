# Sports Coaching from Pose Estimation

#Configuration
Conda environment file: environment.yml





To evaluate on a single image (this is all within the baseline folder): 

python pose_estimation/evaluate.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar --out_file foo

This will save both the image and the raw pose data in the output dir corresponding to the model 
 - foo_pred.jpg
 - foo_pred.jpg.poses.json
 
 To change the image, change the following line in pose_estimation/evaluate.py:
 `eval_image='/home/shared/deepcoaching/custom_data/pro3.jpg' # NOTE: Change this to run on arbitrary image`
 
 