# Predicting media bias of news articles using deep-learning
This repository provides the code I produced for my master thesis with the same title. Necessary packages are stated in the dependencies file. In the following, all steps needed to make use of it and to reproduce results are explained. 

## Data preparation
First the NELA-GT-2018 dataset needs to be downloaded from https://doi.org/10.7910/DVN/ULHLCB, specifically the `articles.db` and `labels.csv` files. `labels.csv` needs to be moved to the `data_preparation` directory (done already). 

Then run the `0_select_news_sources.py` script to save bias labels of sources as well as to create the SQLite command needed to select articles from the `articles.db` file. Use the printed command to delete all articles from the database that are not needed and export the remaining articles to a csv-file (e.g. with the help of SQLite browser https://sqlitebrowser.org/). Save the file as `allsides_articles.csv`  to the `data_preparation/allsides_data` directory.  

Afterwards, the remaining data_preparation scripts can be run in the order of numbering from 1 to 4. To receive also the train set with frequent sentences removed, set the respective variable at the beginning of `1_data_preparation_cleaning_tokenizing.py` to ```True```, adjust the ```affix``` variable at the beginning of files 2 to 4, and run all scripts from 1 to 4 again. 

Note that data preparation code is divided into 4 files for easier memory handling. To run given files 16 GB of RAM are recommended.  
## Deep-learning models
The `bert_model.ipynb` notebook contains the code to train BERT on all datasets and save the resulting metrics. The desired constellation can be selected at the beginning. Besides the augmented datasets also the cost-sensitive version can be chosen. 

The deep-learning benchmark model SHA-BiLSTM (https://arxiv.org/abs/1911.11423) is trained with the `bilstm_benchmark.ipynb` notebook, only on the specific dataset of which news aggregators, tabloids, and frequent sentences are removed at the beginning of the file. 

Both notebooks train models at the end where also the number of run and name affix are chosen. 

## Non-deep-learning model
The `non_deep_learning_model` directory contains all code necessary to prepare the linguistic variables (also for the SemEval dataset after its creation) and run the random forest model. 

At first the `non_dl_benchmark_data_preparation` script needs to be run that creates all linguistic variables except for the part of speech (POS) variables and saves them as numpy arrays. Next the `non_dl_benchmark_pos_variables_preparation.py` file needs to be run to create the remaining variables and save them to numpy files as well. Last, the random forest model can be trained and used for predictions by running the `non_dl_benchmark_model.py` file. 

To do the aforementioned steps for the SemEval dataset, select the semeval affix at the beginning of `non_dl` data preparation files, run them both, and run the `non_dl_semeval.py` file afterwards. 

## Semantic Evaluation 2019 dataset (SemEval)
The SemEval dataset was downloaded from https://doi.org/10.5281/zenodo.1489920, specifically the articles-training-byarticle and ground-truth-training-byarticle files. Then the XML-files were converted to a tsv-file with the help of https://github.com/GateNLP/semeval2019-hyperpartisan-bertha-von-suttner. `semeval_data.tsv` is the resulting file used for all semeval predictions. 

To create all semeval related results, except for the RF predictions explained above, apply the `semeval_results_notebook.ipynb` and (similar to the BERT notebook) select model type and model weights at the beginning of the notebook and run it once for each constellation. 

## LIME and removing  groups of sources from training (other notebooks)
The `lime_notebook.ipynb` in the `other_notebooks` directory creates LIME estimations (https://github.com/marcotcr/lime) of articles, plots them and saves the html texts to file. To choose a specific article select the desired index under the "Selecting article" section of the notebook. Potentially the name of the weights-file needs to be adjusted under "loading weights" to the chosen name during training of BERT. 

The `removed_source_groups_from_training.ipynb` notebook creates prediction results where two groups of sources (one large, one small regarding frequency of articles) with one source per political bias category were removed from the training set. The corresponding model weights are produced with the `bert_model` notebook before and then applied in this notebook. As a result prediction metrics for both groups are given once with each of them included and once excluded from training. No additional parameters need to be adjusted. 

## Other Scripts 
The two remaining python files in `other_scripts` are solely there to produce the tables containing averaged scores of desired figures and two plots presenting the data.    