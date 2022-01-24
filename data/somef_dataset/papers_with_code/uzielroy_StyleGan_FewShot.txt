### Based on # Freeze D for StyleGAN https://arxiv.org/abs/2002.10964

You can run a training session on 30 Monet images using the following Jupiter notebook:
https://drive.google.com/file/d/1UE0WMdxm_CyXLXVd8vMFuZdyl7aLt3Zi/view?usp=sharing

Its recommended that you mount your google drive for easier access to the checkpoint at the end of the training.
Kaggle dataset will be downloaded and saved.
You can view the selected 30 Monet images (done by choosing the farthest 30 points, after dimensionality reduction)

I used the following pre-trained Gan models,https://drive.google.com/file/d/1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ/view
 

### Pre-compute FID activations
```
python precompute_acts.py --dataset DATASET
```

### Run experiments
```
CUDA_VISIBLE_DEVICES=0 python finetune.py --name DATASET_freezeD --mixing --loss r1 --sched --dataset DATASET --freeze_D --feature_loc 3
# Note that feature_loc = 7 - layer_num
``` 

### Evaluation notebook:
https://drive.google.com/file/d/1Nr0Dy5sOFbfKu7aYYekmpt_vpaq9BU4s/view?usp=sharing

Trained for 16 hours.
You can view a sample of 30 generated images and the training graphs (from Tensorboard).
