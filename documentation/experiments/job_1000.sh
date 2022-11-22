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
#SBATCH --output=out-rolf.log
##------------------------ End job description ------------------------

# softsim: name of repo
mkdir /home/u951/u951196/rolf/data/model_1000/

module purge && module load Python/3.10.4-GCCcore-10.3.0 && module load CUDA/11.3.1

source ~/.cache/pypoetry/virtualenvs/rolf-*/bin/activate

srun python3 /home/u951/u951196/rolf/src/bert/train_bert.py

deactivate