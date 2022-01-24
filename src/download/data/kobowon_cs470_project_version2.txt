<h1> Image captioning using BERT and Bottom-up, Top-down Attention </h1>

This is a PyTorch implementation of <a href=http://www.panderson.me/up-down-attention> Bottom-up and Top-down Attention for Image Captioning</a> with <a href=https://arxiv.org/pdf/1810.04805.pdf> BERT</a>. Training and evaluation is done on the MSCOCO Image captioning challenge dataset. Bottom up features for MSCOCO dataset are extracted using Faster R-CNN object detection model trained on Visual Genome dataset. Pretrained bottom-up features are downloaded from <a href =https://github.com/peteanderson80/bottom-up-attention>here</a>. Modifications made to the original model:
<ul>
  <li> ReLU activation instead of Tanh gate in Attention model</li>
  <li> Discriminative supervision in addition to cross-entropy loss</li>
  <li> Add <a href =https://github.com/peteanderson80/bottom-up-attention>BERT</a> instead of embedding layer</li></ul>

<h2> Requirements </h2>

python 3.6<br>
torch 0.4.1<br>
h5py 2.8<br>
tqdm 4.26<br>
nltk 3.3<br>

<h2> Data preparation </h2>

First execute as follow "<b>git clone https://github.com/kobowon/cs470_project_version2.git</b>"

Download the MSCOCO <a target = "_blank" href="http://images.cocodataset.org/zips/train2014.zip">Training</a> (13GB)  and <a href=http://images.cocodataset.org/zips/val2014.zip>Validation</a> (6GB)  images. And put the zip files in <b>"cs470_project_version2/data/coco_2014/"</b> and unzip.

Also download Andrej Karpathy's <a target = "_blank" href=http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip>training, validation, and test splits</a>. This zip file contains the captions. And put the zip file in <b>"cs470_project_version2/data/"</b> and unzip.

<br>


Next, download the <a target = "_blank" href="https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip">bottom up image features</a>.

Unzip the folder and place unzipped folder in <b>"cs470_project_version2/bottom-up_features/"</b> folder.  

<br>

Next type this command in a python 2 environment: 
```bash
python bottom-up_features/tsv.py
```

This command will create the following files (6 files) - train36.hdf5, val36.hdf5,train_ids.pkl,val_ids.pkl,train36_imgid2idx.pkl,val36_imgid2idx.pkl
<ul>
<li>An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an I, 36, 2048 tensor where I is the number of images in the split.</li>
<li>PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.</li>
</ul>

Move these six files to the folder <b>"cs470_project_version2/preprocessed_data/"</b>

<br>

Next, execute script  named <b>'cs470_project_version2/example/download_glue.py'</b> to download glue data to use BERT
```bash
python download_glue.py --data_dir glue_data --tasks all
```

<br>

Next, execute jupyter file named <b>'cs470_project_version2/data/create_final.ipynb'</b> 
```bash
python create_final.ipynb
```
This command will create the following files -

A JSON file for each split containing the order in which to load the bottom up image features so that they are in lockstep with the captions loaded by the dataloader.

<b>File name : TRAIN_GENOME_DETS_preprocessed_coco.json, VAL_GENOME_DETS_preprocessed_coco.json, TEST_GENOME_DETS_preprocessed_coco.json</b>

A cache file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.

<b>File name : cached_TRAIN_Caption, cached_VAL_Caption, cached_TEST_Caption</b>

A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.

<b>File name : TRAIN_CAPLENS_preprocessed_coco.json, VAL_CAPLENS_preprocessed_coco.json, TEST_CAPLENS_preprocessed_coco.json</b>
<br>
***

Next, download pretrained model (checkpoint file) at the google drive link https://drive.google.com/drive/folders/16M4gjlfBWLpySwFoL8eMD_qGUd7EEyZd?usp=sharing <b>dd</b>

ckpt name : BERT_3.pth.tar

and place the checkpoint file in <b>'cs470_project_version2/ckpt/'</b>


Next, download java 1.8 and <a target = "_blank" href=https://github.com/poojahira/image-captioning-bottom-up-top-down/tree/master/nlg-eval-master>nlg_eval_master folder</a> and place this file on <b>'cs470_project_version2/experiment/'</b> folder and type the following two commands at the folder position in command:
```bash
pip install -e .
nlg-eval --setup
```
This will install all the files needed for evaluation.
***



<h2> Training </h2>

<b>You can check A@5 performance during training :)</b>

To train the bottom-up top down model from scratch, go to <b>'cs470_project_version2/'</b> folder and execute 'model_training.ipynb':

The dataset used for learning and evaluation is the MSCOCO Image captioning challenge dataset. It is split into training, validation and test sets using the popular Karpathy splits. This split contains 113,287 training images with five captions each, and 5K images respectively for validation and testing. Teacher forcing is used to aid convergence during training. Teacher forcing is a method of training sequence based tasks on recurrent neural networks by using the actual or expected output from the training dataset at the current time step y(t) as input in the next time step X(t+1), rather than the output generated by the network. Teacher forcing addresses slow convergence and instability when training recurrent networks that use model output from a prior time step as an input.

Weight normalization was found to prevent the model from overfitting and is used liberally for all fully connected layers.

Gradients are clipped during training to prevent gradient explosion that is not uncommon with LSTMs.

The attention dimensions, and hidden dimensions of the LSTMs are set to <b>384 because of resouce limitation</b> (<a target = "_blank" href=https://github.com/poojahira/image-captioning-bottom-up-top-down/>original</a> : batch is 1024).

The word embedding dimension is set to 768 for BERT (<a target = "_blank" href=https://github.com/poojahira/image-captioning-bottom-up-top-down/>original</a> is 1024)


Dropout is set to 0.5. Batch size is set to <b>20 because of resouce limitation</b> (<a target = "_blank" href=https://github.com/poojahira/image-captioning-bottom-up-top-down/>original</a> : batch is 100). 36 pretrained bottom-up feature maps per image are used as input to the Top-down Attention model. The Adamax optimizer is used with a learning rate of 2e-3. Early stopping is employed if the BLEU-4 score of the validation set shows no improvement over 20 epochs.

***
<h2> Evaluation </h2>

To evaluate the model on the karpathy test split, edit the <b>'cs470_project_version2/model_evaluation.ipynb'</b> file to include the model checkpoint location

and execute 'model_training.ipynb':

Beam search is used to generate captions during evaluation. Beam search iteratively considers the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keeps only the resulting best k of them. A beam search of five is used for inference.

The metrics reported are ones used most often in relation to image captioning and include BLEU-4, CIDEr, METEOR and ROUGE-L. Official MSCOCO evaluation scripts are used for measuring these scores.

***
<h2>References</h2>

Code adapted with thanks from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, 

Basic code adapted with thanks from https://github.com/poojahira/image-captioning-bottom-up-top-down/, 

Bert basic code adapted with thanks from https://github.com/huggingface/transformers

Evaluation code adapted from https://github.com/Maluuba/nlg-eval/tree/master/nlgeval

Tips for improving model performance and code for converting bottom-up features tsv file to hdf5 files sourced from https://github.com/hengyuan-hu/bottom-up-attention-vqa

https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

