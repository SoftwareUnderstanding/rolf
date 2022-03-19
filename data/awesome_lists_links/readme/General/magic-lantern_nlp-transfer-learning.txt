# nlp-transfer-learning

Clinical Natural Language Processing Transfer Learning based on ULMFit.

For details on ULMFit, see:

* [Introducing state of the art text classification with universal language models](http://nlp.fast.ai/) - Blog Post
* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) - arXiv paper; also peer reviewed and accepted for the 2018 Annual Meeting of the Association for Computational Linguistics.

## Data

This project uses data from [MIMIC-III](https://mimic.physionet.org/). The data is freely available but does require pre-registration and some training before access will be granted.

Once you have access, you can download the .gz version of the data files with these commands - replace `mimicusername` with your actual username:

    wget --user mimicusername --ask-password https://physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_4/NOTEEVENTS.csv.gz
    wget --user mimicusername --ask-password https://physionet.org/works/MIMICIIIClinicalDatabase/files/version_1_4/ADMISSIONS.csv.gz
    gunzip *.csv.gz

In the Python Jupyter notebooks, the data is assumed to be located in the directory `Path.home()/'mimic'`

## How to setup environment

    conda create -y -n fastai python=3.6
    conda activate fastai
    pip install dataclasses gpustat
    conda install -y -c pytorch pytorch torchvision cudatoolkit=9.0
    conda install -y -c fastai fastai
    conda install -y ipykernel nbconvert ipywidgets scikit-learn
    conda install -y -c conda-forge jupytext
    conda install -y -c conda-forge jupyterlab
    conda install -y -c conda-forge altair vega_datasets


*Project Structure and Organization based off https://github.com/callahantiff/Abra-Collaboratory/*
