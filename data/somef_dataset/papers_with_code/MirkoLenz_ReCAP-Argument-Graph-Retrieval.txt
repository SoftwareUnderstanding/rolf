# ReCAP: Argument Graph Retrieval

[![DOI](https://zenodo.org/badge/192173055.svg)](https://zenodo.org/badge/latestdoi/192173055)

This program has been used to perform the evaluation for my Bachelor's Thesis.
It provides a retrieval for argumentation graphs.

## System Requirements

- Docker and Docker-Compose

## Installation

### Application

Duplicate the file `config_example.yml` to `config.yml` and adapt the settings to your liking.
Please do not edit the webserver settings as Docker depends on them.


### Embeddings

The following list contains all models used in the paper together with instructions to make them usable for the software.
It is recommended to rename the files to a memorable name and put them in a folder named `data/embeddings`.

- [Google Word2Vec:](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  - Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J.: Distributed Representations of Words and Phrases and their Compositionality (2013), <https://arxiv.org/abs/1310.4546>
  - `docker-compose run --rm app python -m recap_agr.cli.convert bytes-text path/to/GoogleNews-vectors-negative300.bin.gz`
  - `docker-compose run --rm app python -m recap_agr.cli.convert model-gensim path/to/GoogleNews-vectors-negative300.txt`
- Custom Doc2Vec: Not yet available.
- [fastText:](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)
  - Bojanowski, P., Grave, E., Joulin, A., Mikolov, T.: Enriching Word Vectors with Subword Information (2016), <https://arxiv.org/abs/1607.04606>
  - Unpack the file.
  - `docker-compose run --rm app python -m recap_agr.cli.convert model-gensim path/to/crawl-300d-2M.vec`
- [GloVe:](http://nlp.stanford.edu/data/glove.840B.300d.zip)
  - Pennington, J., Socher, R., Manning, C.: Glove: Global Vectors for Word Representation. In: Proceedings of EMNLP (2014). <https://doi.org/10.3115/v1/D14-1162>
  - Unpack the file.
  - Run `cat path/to/glove.6B.300d.txt | wc -l` to obtain the number of items.
  - Add `#LINES 300` as the first line of the file, e.g. `1000 300` if the output above gave 1000 (recommended to use `vim`).
  - `docker-compose run --rm app python -m recap_agr.cli.convert model-gensim path/to/glove.6B.300d.txt`
- [Infersent:](https://dl.fbaipublicfiles.com/infersent/infersent1.pkl)
  - Conneau, A., Kiela, D., Schwenk, H., Barrault, L., Bordes, A.: Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (2017), <https://arxiv.org/abs/1705.02364>
  - No modification needed.
- [USE-D:](https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed)
  - Cer, D., Yang, Y., Kong, S.y., Hua, N., Limtiaco, N., John, R.S., Constant, N., Guajardo-Cespedes, M., Yuan, S., Tar, C., Sung, Y.H., Strope, B., Kurzweil, R.: Universal Sentence Encoder (2018), <http://arxiv.org/abs/1803.11175>
  - Unpack the file.
- [USE-T:](https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed)
  - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems 30, pp. 5998–6008 (2017)
  - Unpack the file.




## Usage

It is possible to run the software with

```docker-compose up```

This will download all required data on the first run and thus may take a while.
Future runs are cached and the app available immediately.

The webserver is then accessible on <http://localhost:8888>



## Data Folder Contents

The following folders need to be specified:

- `casebase_folder`
- `queries_folder`
- `embeddings_folder`
- `candidates_folder`
- `results_folder`

### Case-Base and Queries

All files need to be present in the AIF- or OVA-format (and thus be `.json` files).


### Embeddings

Only the native `gensim` format is supported.


### Results

No file needs to be put in here.
The exporter will write the results to this folder.
However, the folder needs to be created manually.


### Candidates

For each query, a candidates file with the following content has to be provided so that the evaluation metrics are calculated.

_Please note:_ Candidates and rankings do not need to contain the same filenames.

```json
{
	"candidates": [
		"nodeset6366.json",
		"nodeset6383.json",
		"nodeset6387.json",
		"nodeset6391.json",
		"nodeset6450.json",
		"nodeset6453.json",
		"nodeset6464.json",
		"nodeset6469.json"
	],
	"rankings": {
		"nodeset6366.json": 2,
		"nodeset6383.json": 2,
		"nodeset6387.json": 3,
		"nodeset6391.json": 2,
		"nodeset6450.json": 2,
		"nodeset6453.json": 2,
		"nodeset6464.json": 2,
		"nodeset6469.json": 1
	}
}

```


## Important Information

- [License](LICENSE)
- [Copyright](NOTICE.md)
