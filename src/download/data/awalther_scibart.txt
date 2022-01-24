# scibart

So far, this is a refactoring job of a notebook in [CurationCorp](https://github.com/CurationCorp)'s amazing [curation-corpus repository](https://github.com/CurationCorp/curation-corpus) for training on GPU clusters to tune [BART](https://arxiv.org/abs/1910.13461) for abstractive summarization of scientific literature.

Part of the [Coronawhy](http://coronawhy.com) project.


## How to create a dataset from scratch

Currently, the dataset is sourced as follows:

* Text-abstract pairs from Arxiv and the [Semantic Scholar Corpus](http://s2-public-api-prod.us-west-2.elasticbeanstalk.com/corpus/) as provided by [Santosh-Gupta](https://github.com/Santosh-Gupta)'s [ScientificSummarizationDataSets](https://github.com/Santosh-Gupta/ScientificSummarizationDataSets) repo
* Text-headline pairs from WikiHow, provided by [mahnazkoupaee](https://github.com/mahnazkoupaee)'s [WikiHow-Dataset](https://github.com/mahnazkoupaee/WikiHow-Dataset) repo
* [Curation Corpus](https://github.com/CurationCorp/curation-corpus)

To create a new dataset from scratch:

1. Download the ArXiv and Semantic Scholar Corpus datasets from gdrive (as described [here](https://github.com/Santosh-Gupta/ScientificSummarizationDataSets)) and unzip into `raw_data/ArxivStructuredAbstractSectionalSummaries` and `raw_data/SemanticScholarAbstractSectionSummaryDataSet`
2. Download wikihowAll.csv (as described [here](https://github.com/mahnazkoupaee/WikiHow-Dataset)) into `raw_data/wikihow`
3. Scrape the Curation Corpus dataset as explained in the [repo](https://github.com/CurationCorp/curation-corpus), then move `curation-corpus-base-with-articles.csv` to `raw_data/curation_corpus`
4. Run `python src/data/create_dataset.py`. This will create a new folder called `data` with ~40 compressed parquet files

The current dataset is stored in a single pandas dataframe with the following schema:

| Column name        | Column Type           | Description  |
| ------------- |:-------------:| -----:|
| text      | str |  Original text on which the summary is based
| summary      | str      |   Summary of the original text |
| data_src | str      |    Directory name of the original dataset in `raw_data` |
