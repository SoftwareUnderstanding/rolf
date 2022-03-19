# HPCAI-2020-BERT-Submission

## Introduction
This codebase is to reproduce the results in the report submitted to APAC HPCAI 2020. We optimize the distributed performance by using gradient checkpointing. The baseline is run on 1 V100 GPU on NSCC DGX and the optimized code is run on 2 nodes with 4 GPUs on each node on NSCC DGX. The optimized code can achieve more than 8 times throughput compared to baseline experiment.


## Gradient checkpointing
Gradient checkpointing is a method to save GPU memory and boost the batch size in expense of some computation time. The paper is published here https://arxiv.org/abs/1604.06174. This method does not modify the structure of the model, it can be integrated into the code seamlessly. The original code is in `modeling_v0.py` and the code with gradient checkpointing is in `modeling_v2.py`.

## Checkpoint saving
This code will save checkpoints at 20 min automatically as configured by `save_checkpoints_secs` in the TensorFlow estimator:
```python
run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir if master_process else None,
        session_config=config,
        save_checkpoints_secs=60*20 if master_process else None,
        log_step_count_steps=FLAGS.display_loss_steps,
        keep_checkpoint_max=1)
```

This ensures the experiments are compared consistently.

## Set up
```shell
git clone https://github.com/FrankLeeeee/HPCAI-2020-BERT-Submission.git
```

## Dependency
Singularity image: nvcr.io/nvidia/tensorflow:20.02-tf1-py3.sif

CUDA | cuDNN | NCCL | Tensorflow | Horovod | Python
--- | - | - | - | - | - 
10.2 | 7.6.5 | 2.5.6 | 1.15 | 0.19 | 3.6.9

## Results in 20mins
model | bs | xla | #gpus | grad ckpt | optim | #steps/sec | #sentences/sec | #steps at 20mis | throughput at 20mins | acc | loss
--- | - | - | - | - | - | - | - | - | - | - | -
Baseline | 24 | True | 1 | False | LAMB | 4.8 | 115.2 | 2938 | 70512 | 0.804 | 0.504
Optimized | 48 | True | 2x4 | True | LAMB | 2.53 | 971.52 | 1225 | 470400 | 0.8641 | 0.3703
* #sentences/sec = bs * #steps/sec * #gpus
* throughput at 20mins = bs * #steps at 20mis * #gpus

**The optimized code is running on 2 nodes with each node having 4 GPUs, making it 8 GPUs. This is the max resources we successfully queued for**

## Run Baseline

To run the baseline experiment, you need to follow the following steps:

1. change the variables `BERT_DIR`, `GLUE_DIR` and `RESULTS_DIR` in `hpcai_scripts/run_glue.sh`

2. The baseline is using the original model implementation provided by Nvidia. Thus, you need to edit the `run_classifier.py` like below:
```shell
# change line 32
# import modeling_v2 as modeling 
# to the line below 
import modeling_v0 as modeling
``` 

3. Also, make sure the config is consistent with the baseline config in the `run_glue.sh`
```shell
task_name=${1:-"MNLI"}
batch_size=${2:-"24"}
learning_rate=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
num_gpu=${6:-"1"}
seq_length=${7:-"128"}
doc_stride=${8:-"64"}
epochs=${9:-"3.0"}
ws=${10:-"0.1"}
init_checkpoint=${11:-"$BERT_DIR/bert_model.ckpt"}
```

4. get interactive job
```
qsub -I -q dgx -l walltime=1:00:00,select=1:ngpus=1:ncpus=5:mpiprocs=1 -P $ProjectID 
```

5. run the script
```shell
cd ./HPCAI-2020-BERT-Submission/BERT
singularity exec /home/projects/ai/singularity/nvcr.io/nvidia/tensorflow:20.02-tf1-py3.sif ./hpcai_scripts/run_glue.sh
```
6. change the variables in `eval.sh` and run it
```shell
bash ./hpcai_scripts/eval.sh
```

## Run Optimized Code

To run the optimized code on 2 nodes with 4 GPUs on each node, you need to follow the following steps:
1. Copy folders `apps` and `scripts` under the $PATH_TO_SUBMISSION/multinode_communication to $HOME directory, and make a empty folder `sshcont` under $HOME directory
```shell 
cp -r $PATH_TO_SUBMISSION/multinode_communication/* $HOME
mkdir $HOME/sshcont
```

2. Edit file $HOME/scripts/sshcont/`job_tensorflow_gloo.sh`
```shell
# edit line 6: RESULTS_DIR pointing to the directory where experiment results are saved 
RESULTS_DIR=...
# edit line 13: GLUE_SCRIPT pointing to `BERT/hpcai_scripts/run_glue_nscc.sh`
GLUE_SCRIPT=$PATH_TO_SUBMISSION/BERT/hpcai_scripts/run_glue_nscc.sh
```
3. Edit file $HOME/scripts/sshcont/`tensorflow.sh` (this is the time for establishing connections among all nodes, and usually 180s is enough. you may increase it as you scale to more nodes or the network is slower) 
```shell
# edit line 13 and 14: xxxs (e.g. 180s) 
echo "Waiting xxxs for SSH servers to be up"
sleep xxxs
```

4. The optimized code is using the model with gradient checkpointing. Thus, you need to edit the `run_classifier.py` like below:
```shell
# change line 32
# import modeling_v0 as modeling 
# to the line below
import modeling_v2 as modeling
``` 

5. Also, make sure the config is consistent with the optimized config in the `run_glue_nscc.sh`
```shell
task_name=${1:-"MNLI"}
batch_size=${2:-"48"}
learning_rate=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
num_gpu=${6:-"1"} 
seq_length=${7:-"128"}
doc_stride=${8:-"64"}
epochs=${9:-"3.0"}
ws=${10:-"0.1"}

# NOTE
# variable num_gpu is not in effect in the optimized code
```

6. Set the correct `BERT_DIR`, `GLUE_DIR` and `run_classifier.py` path in `run_glue_nscc.sh`. Set the correct `RESULTS_DIR` in `job_tensorflow_gloo.sh` 

7. get interactive job
```shell
qsub -I -q dgx -l walltime=1:00:00,select=2:ngpus=4:ncpus=20:mpiprocs=4,place=scatter -P $ProjectID 
```

8. run experiment 
```shell
bash $HOME/scripts/sshcont/invocation
```

9. run evaluation. As gradient checkpointing is not needed in evaluation, thus, you need to comment out the lines 259, 1041, 1042, 1067, 1068, 1088, 1089 and change `ckpt` to be `False` in line 1018.
```shell
# comment out the following lines
# 259: pool_output = tf.contrib.layers.recompute_grad(pool_output)
# 1041: dense_output = tf.contrib.layers.recompute_grad(
# 1042:                           dense_output)
# 1067: dense_intermediate = tf.contrib.layers.recompute_grad(
# 1068:                         dense_intermediate)
# 1088: dense_output_2 = tf.contrib.layers.recompute_grad(
# 1089:                         dense_output_2)

# change line 1018
# ckpt=layer_idx%2==0) -> ckpt=False)

# run evaluation
bash $PATH_TO_SUBMISSION/BERT/hpcai_scripts/eval.sh
```
