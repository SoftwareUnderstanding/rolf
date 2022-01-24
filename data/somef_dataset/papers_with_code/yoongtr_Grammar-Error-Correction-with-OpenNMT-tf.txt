# Grammar-Error-Correction-with-OpenNMT-tf
Grammar Error Correction with OpenNMT-tf using Neural Machine Translation and Transformer model

# OpenNMT
Model used: transformer
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. arXiv 2017. arXiv preprint arXiv:1706.03762.
https://arxiv.org/abs/1706.03762

Use python 3.7.3 for underthesea compatibility inside notebook
Use GPU (ONMT-tf will automatically detect GPU and assign jobs)
GPU installation within conda env: `conda create --name tf_gpu tensorflow-gpu`

## 1) Install OpenNMT-tf
`pip3 install OpenNMT-tf`

## 2) Prepare data / Preprocessing
`cd preprocess`

* Requirements
    * underthesea - a VN NLP toolkit:
    `pip3 install underthesea`
    * Use Python 3.7 for underthesea. Python 3.8 causes some problems with import
   
* Preprocess raw csv files using preprocess_vn.ipynb

* Output files for ONMT: src files are "wrong" data, tgt files are "correct" data
    * src-train.txt
    * src-val.txt
    * src-test.txt
    * tgt-train.txt
    * tgt-val.txt
    * tgt-test.txt
    
* Move all above files to ~/training folder    


## 3) Build vocabulary
* This creates the vocabulary for the model. Build vocabulary from data files by:
   `onmt-build-vocab --size 10000 --save_vocab src-vocab.txt src-train.txt`
   `onmt-build-vocab --size 10000 --save_vocab tgt-vocab.txt tgt-train.txt`
* Alternatively, vocabulary can also be built from external sources, e.g. Wikidump. If data is highly specific on one subject, build from data files for higher accuracy
* Change vocabulary --size to fit your data size

## 4) Create config data.yml file (in ~/training)
See [OpenNMT parameters documentation](https://opennmt.net/OpenNMT-tf/configuration.html) for parameters tuning

## 5) Train model
` cd training`
* Parameters
    * To specify which GPU to run: CUDA_VISIBLE_DEVICES=gpu_id_1,gpu_id_2
    * To choose number of GPUs to train (batches are processed in parralel): --num_gpus no_of_gpus

* Specific GPU `CUDA_VISIBLE_DEVICES=1 onmt-main --model_type Transformer --config config.yml --auto_config train --with_eval` 
* Multiple GPU `CUDA_VISIBLE_DEVICES=1,2 onmt-main --model_type Transformer --config config.yml --auto_config train --with_eval --num_gpus 2`

* Run on CPU `onmt-main --model_type Transformer --config config.yml --auto_config train --with_eval`

Track logs: `tensorboard --logdir="."`

CUDA_VISIBLE_DEVICES for specifying the GPU tf will see

## 6) Translate
`CUDA_VISIBLE_DEVICES=1,2 onmt-main --config config.yml --auto_config infer --features_file src-test.txt --predictions_file predictions.txt`

Predictions are saved inside predictions.txt (change file name accordingly)

## 7) Evaluate with BLEU score
Move predictions.txt and tgt-test.txt to smooth_Bleu folder

`cd ~/smooth_Bleu`
`python3 bleu.py -r tgt-test.txt -t predictions.txt`

OR can include BLEU in YAML file (TBC)
