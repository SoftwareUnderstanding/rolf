#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100
#SBATCH --job-name=softsim
#SBATCH --mem-per-cpu=64000
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.garijo@upm.es
#SBATCH --output=out-%j.log
##------------------------ End job description ------------------------

mkdir /home/u951/u951196/softsim/data/model_1000/

module purge && module load Python/3.9.5-GCCcore-10.3.0 && module load CUDA/11.3.1

source /home/u951/u951196/softsim/torch_cuda/bin/activate

srun python3 /home/u951/u951196/softsim/src/SimGNN_cuda/test.py --data-path /home/u951/u951196/softsim/data/SoftwareSim/post_process/ --json-path /home/u951/u951196/softsim/data/SoftwareSim/final_data/ --score-path /home/u951/u951196/softsim/data/SoftwareSim/training.csv --save-path /home/u951/u951196/softsim/data/model_1000/ --epochs 10 --sim_type sbert --func "1000"

deactivate