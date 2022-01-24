# Individual Level NLP Classifier
Individual level NLP classifier is a consulting project worked with Basilica.ai. Basilica.ai is a B2B startup that providing sentiment as a service for social media companies. The goal is to have individual level classifiers so that companies can sort newly generated social media posts based on individual user's interest.

[Here](https://docs.google.com/presentation/d/1sYz4yeRL0rFM4c1ScgJoM6id3RIu6Fh1wfRQ1_rghFA/edit#slide=id.p) are the slides for the project


## Project:

- **PSM_nlp** : All source code for production within structured directory

- **demo_data** :  Sample data for demo purpose. It is one real dataset from Basilica and removed their feature embedding.

- **jupyter_notebooks** : Contains all the jupyter notebooks for building up this project, including EDA, ULMFit / BERT training and post PCA / Siamese network analysis

## Setup
Clone repository and update python path
```
cd ~
git clone https://github.com/heye0507/individualNLPClassifier.git
cd individualNLPClassifier/
docker build -t nlp_classifier .
docker run -it nlp_classifier
```


## Demo
Run the following two lines for demo

- Take Basilica dataset(s)
- Using AWD_LSTM pre-trained general classification model, fine tune classification head on new dataset
- Save the Pytorch model for in the designated path
```
source activate nlp_model_gen
python3 runner.py --input-file=demo_data/
```
## Package

PSM_nlp can be used as a python package

- PSM_nlp.bert_interface and PSM_npl.bert_trainer contains BERT pre-processing and training API
- PSM_nlp.interface and PSM_nlp.trainer contains AWD_LSTM pre-processing and training API

Here is an example of how to build both trainer for BERT and AWD_LSTM

```
python3 setup.py build

import PSM_nlp
import os
from PSM_nlp.bert_interface import *
from PSM_nlp.bert_trainer import *

path = Path(os.getcwd() + '/demo_data')
bert_interface = BERT_Interface('bert-base-uncased', path.ls())
bert_interface.pre_processing()
bert_trainer = BERT_Trainer(bert_interface)

### Please note you will need GPU to run the following line ###
bert_trainer.train_individual_clasifier()


### For AWD_LSTM model ###
import PSM_nlp
import os
from PSM_nlp.interface import *
from PSM_nlp.trainer import *
from PSM_nlp.downloader import *

path = Path(os.getcwd() + '/demo_data')
download_file_from_google_drive(data_lm_large_id,path/'data_lm_large.pkl')
download_file_from_google_drive(model_id,path/'general-clasifier-0.84.pth')


interface = Interface(path.ls(),eval_mode=True)
model_path = path/'models'
interface.pre_processing(lm_path=model_path) 
trainer = Trainer(interface,model_path)
trainer.train_individual_clasifier()
```


## Model Apporach 
- Baseline BERT, build classification head on BERT
- ULMFit, using AWD_LSTM as encoder (language model), build classification head on top of fine-tuned language model https://arxiv.org/abs/1801.06146

## Input format
- For individual classification model generation, store all the datasets into demo_data/ or create new folder under /individualNLPClassifier/your_data_folder
- The dataset should be in JSON format
- Each JSON file is an object with the fields id, num_data_points, and data_points. data_points is an array of objects with the fields body, embedding, and source. body is the original text of the message. source is the site the message came from. embedding is the embedding produced language model (For non-Basilica dataset, simply make it empty list [], Please see demo.json in demo_data/ folder for more details) 

## Future Work
- Intergrate AWD_LSTM language model fine-tuning API. 
- Intergrate BERT language model fine-tuning API.


