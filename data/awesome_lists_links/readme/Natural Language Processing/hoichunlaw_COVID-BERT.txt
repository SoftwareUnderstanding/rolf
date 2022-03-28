# COVID-BERT: Pre-training of Deep Bidirectional Transformers for COVID-19 Text Understanding

## Abstract
Fine-tuned BERT (https://arxiv.org/abs/1810.04805) on published medical research paper (only abstract section). Research paper available from Kaggle. (https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

## Model URL
https://saved-language-models.s3.amazonaws.com/covid-bert-base-uncased.zip

## Usage
```
from transformers import AutoTokenizer, AutoModel
covid_tokenizer = AutoTokenizer.from_pretrained(your_model_path)
covid_model = AutoModel.from_pretrained(your_model_path)
```