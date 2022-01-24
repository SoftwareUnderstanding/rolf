# Retrosynthesis prediction using Transformers

The following instructions guide you in reproducing results from [1,2] using the powerful Transformers architecture [3].

[1] https://link.springer.com/chapter/10.1007/978-3-030-30493-5_78 \
[2] https://www.nature.com/articles/s41467-020-19266-y \
[3] https://arxiv.org/abs/1706.03762

## Requirements
This repository uses ```Python 3.8.11``` and requires some libraries to run. Please ensure that you have rdkit ```pip install rdkit```.\
PyTorch 1.9 with cuda 11.1 was used, and the trainer writes to a tensorboard, and can be installed as follows:
```bash
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard
```
```pip install tqdm``` is also required.

## Installation
```bash
git clone https://github.com/ArvidWartenberg/retrosynthesis.git
cd retrosynthesis/retrosynthesis_cluster_training/
```

## Train your model
Choose any name for your run, if not it will default to a "hh/mm/ss".\
The key model parameters can easily be passed as arguments. See run.py for detailed descriptions.
```bash
python3 run.py --device 0 --name my_run
```

## Inference
Differnt weight checkpoints are made during training. These & the used model params lie in the models/chemformers folder.\
The specific path to your model will be /yyyy/mm/dd/my_name. The key arguments for inference are listed below
```python
parser.add_argument("--model", help="date/time model indicator", required=False)
parser.add_argument("--algorithm", help="search algorithm", default="greedy", required=False)
parser.add_argument("--device", help="choose device", default="0", required=False)
parser.add_argument("--weights", help="Choose checkpoint", default="best_acc", required=False)
parser.add_argument("--dataset", help="choose dataset", default="non-augmented", required=False)
parser.add_argument("--n_infer", help="num inference points", default="all", required=False)
parser.add_argument("--k", help="top k to be reported", default=5, required=False)
parser.add_argument("--beam_size", help="beam size", default=5, required=False)
````
Example: Run inference on entire test/val (using same xD) using greedy decoding. Model with best token ACC during training is used
```bash
python3 infer.py --model yyyy-mm-dd/my_run --weights best_acc --n_infer all --device=0 --algorithm greedy
```
Inferred reactants will be written to a pickle file (script prints exact path).\
Also, each inference run writes to the RESULTS.txt file in root with a short summary of accuracies & which model was used.

## License
You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit.
