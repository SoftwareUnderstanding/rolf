# Named Entity Recognition

![GitHub last commit](https://img.shields.io/github/last-commit/blawok/named-entity-recognition)
![GitHub repo size](https://img.shields.io/github/repo-size/blawok/named-entity-recognition)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 


Training and deployment of BiLSTM and RoBERTa in AWS SageMaker for NER task.  
I strongly encourage you to take advantage of [Jupyter Notebook Viewer](https://nbviewer.jupyter.org/github/blawok/named-entity-recognition/tree/master/) to explore this repository.

## tl;dr
Fine-tuned RoBERTa (F1 0.838) turned out to outperform BiLSTM (F1 0.788). In this repository you can explore the capabilities of AWS SageMaker (training and deployment scripts for Tensorflow and PyTorch), S3, Lambdas, API Gateway (model deployment) and Flask Framework (Web App).

## Project report
If you would like to go through the whole project you can start with the [project report](https://github.com/blawok/named-entity-recognition/blob/master/reports/project_report.pdf) and then follow the codes as it is described in the section below.

## Project flow

![mermaid_flowchart](https://user-images.githubusercontent.com/41793223/83953788-abfd4980-a843-11ea-95a2-613508068499.JPG)

If you would like to replicate the results simply follow the flowchart - you will find all necessary scripts in [src](https://github.com/blawok/named-entity-recognition/tree/master/src).


## Data
Data source: [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)  

This is the extract from Groningen Meaning Bank (GMB) corpus which is tagged, annotated and
built specifically to train the classifier to predict named entities such as location names, organisations, time, people identification etc.

Dataset consits of:  
● 47959 sentences  
● 1354149 words  
● 17 distinct entity tags  

## BiLSTM and RoBERTa Source
[src](https://github.com/blawok/named-entity-recognition/tree/master/src) directory contains source code for both models, but also EDA,  full data preparation and inference code. I tried to keep it together with [cookie-cutter](https://github.com/drivendata/cookiecutter-data-science), but had to make some slight adjustments. Training processes are thoroughly described in train_*.ipynb notebooks.

 Folder tree made with simple yet amazing [repository tree](https://github.com/xiaoluoboding/repository-tree) :

```
├─ src
│  ├─ data_processing
│  │  ├─ helpers.py
│  │  ├─ prepare_data_bilstm.ipynb
│  │  └─ prepare_data_for_roberta.ipynb
│  ├─ eda
│  │  └─ eda.ipynb
│  ├─ serve
│  │  ├─ predict.py
│  │  └─ requirements.txt
│  ├─ source_bilstm
│  │  └─ train_bilstm.py
│  ├─ source_roberta
│  │  ├─ requirements.txt
│  │  ├─ train_roberta.py
│  │  └─ utils.py
```
> I also [experimented](https://github.com/blawok/named-entity-recognition/tree/master/experiments) with different architectures such as BERT, DistilBERT and BiLSTM-CRF (which unfortunately is not yet supported in AWS SageMaker using TensorflowPredictor and script mode). However, RoBERTa seemed to perform better than all of them, I am curious how it will compare to BiLSTM-CRF.

## Model evaluation

Both models were tested on the same test set (10%) and achieved following results:

|                |F1 Score                          |
|----------------|-------------------------------|
|BiLSTM|`0.788   `         |
|RoBERTa          |`0.838`            |

Fine-tuned RoBERTa clearly outperforms BiLSTM, as well as all models presented in [kaggle kernels](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/kernels) for this dataset.

## Model Deployment
For this purpose I utilized two additional AWS services: Lambda and API Gateway. I also developed [Flask WebApp](https://github.com/blawok/named-entity-recognition/tree/master/web_app) that enables the user to use the API. 

If you will need any help with Lambda or API Gateway check out this [deployment cheatsheet](https://github.com/blawok/named-entity-recognition/blob/master/src/utils/sagemaker_deployment_cheatsheet.md).

Brief look into the app: 

![Recordit GIF](http://g.recordit.co/jiRqtNxMYD.gif)

## Further research
- [ ] Experiment with CRF layers (combined with BiLSTM and some embeddings like ELMo)
- [ ] Experiment with CNN character embeddings
- [ ] Experiment with different XAI techniques to explain NER predictions (like LIME, eli5)

## References
- https://arxiv.org/abs/1907.11692  
- https://huggingface.co/transformers/model_doc/roberta.html  
- https://github.com/huggingface/transformers/pull/1275/files  
- https://github.com/huggingface/transformers/tree/master/examples/token-classification  
- https://www.kaggle.com/debanga/huggingface-tokenizers-cheat-sheet  
- https://github.com/billpku/NLP_In_Action  
- https://androidkt.com/name-entity-recognition-with-bert-in-tensorflow/  
- https://github.com/smart-patrol/sagemaker-bert  
- https://www.depends-on-the-definition.com/tags/named-entity-recognition/

## Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.  
If you would like to collaborate on points from **further research**, feel free to open an issue or msg me on [linkedin](https://www.linkedin.com/in/bkowalczuk/) :wink:



