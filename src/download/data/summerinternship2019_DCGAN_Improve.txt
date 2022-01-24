# DCGAN_Improve
Goal 1 Improved
Goal 1er.

To develop a system implementing Generative Adversarial Networks (GANs) 
to generate alternate versions of videogame/anime characters in 2D.

 

                Example: Generating alternate versions of Videogame characters = new character

               

Given: All characters → new character
referred to: https://arxiv.org/abs/1511.06434

## Dataset
download form google drive Pokemon 819 iamges: 
https://drive.google.com/open?id=1HwJuvMUQzxK0jduvnHhUTH_0Lv2jzYt0
## Pre-processing
reformat and resize iamges
###usage 
libraries PIL, os, sys
example :  
         [>> python resizejpg2png.py source_dir destination_dir image_size]
         
         >> python resizejpg2png.py data_int  data72png  72 
## Prerequisites
Python 3.6.3 |Anaconda, Inc.| (default, Oct 13 2017, 12:02:49) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

create anaconda environment

    conda env create -f environment.yml
    source activate tfpgu
## Usage
### local run
   `>> python pokemon_v1.py`
### batch run
Edit gpu node
  #SBATCH --nodelist compute024
  #SBATCH --partition=gpu
  ##SBATCH --gres=gpu:0  
base in 'sinfo' cluster information
 
    > sbatch run_gpu_slurm

## Result

![](https://github.com/summerinternship2019/DCGAN_Improve/blob/master/poke72.gif)
