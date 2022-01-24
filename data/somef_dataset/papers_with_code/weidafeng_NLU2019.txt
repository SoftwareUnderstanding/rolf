# NLU2019
NLU2019 project: Question NLI. The task is to determine whether the context sentence contains the answer to the question (entailment or not entailment).
![IMAGE.png](NLU2019-project.png) 

## Usage:
1. Download dataset.
```bash
$ python download_glue_data.py --data_dir glue_data --tasks all
```
This code borrowed from [here](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e), you need using VPN to run it, or you can using my provided 'glue_data.zip' easily.

2. Install `apex`.
`apex` is a pyTorch extension: Tools for easy mixed precision and distributed training in Pytorch. The official repository is [here](https://github.com/NVIDIA/apex).
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
3. Install the necessary libary `pytorch-pretrained-bert`. 
```bash
$ pip install pytorch-pretrained-bert
```

4. Clone this repository.
```bash
$ git clone https://github.com/weidafeng/NLU2019.git  
$ cd NLU2019
```

5. Train. You will get the pretrained model flies('config.json  eval_results.txt  pytorch_model.bin  vocab.txt') in `glue_data/QNLI/eval_result`. 
```bash
$ bash train.sh
```
Here is my results:
```
acc = 0.9110378912685337
eval_loss = 0.501230152572013
global_step = 16370
loss = 0.0006768958065624673
```

6. Predict. You will load the pretrained model to predict, and get the submission `QNLI.tsv` in  `glue_data/QNLI/eval_result`.
```bash
$ bash test.sh
```

7. Submission. Create a zip of the prediction TSVs, without any subfolders, e.g. using:
```bash
$ zip -r submission.zip *.tsv
```
Here is my glue result:
![glue_result](GLUE_RESULTS.png)
Trained model is too big to store in GitHub, if needed, please feel free to contact me.  

## File path tree and annotations:
```
├─bert-base-uncased # path to store the cached pretraind `bert` model(automatically download from s3 link)
├─glue_data  
│  └─QNLI 	# path to sort GLUE data 
│      └─results 	# path to store trained model('config.json  eval_results.txt  pytorch_model.bin  vocab.txt') and the prediction results(`QNLI.tsv`)
├─model 	# main code for this project
└─submission 	#  submission file
```

## Reference.
1. https://github.com/google-research/bert
2. https://github.com/huggingface/pytorch-pretrained-BERT
3. https://arxiv.org/abs/1810.04805
4. https://gluebenchmark.com/faq