# Categorical Modularity: A Tool For Evaluating Word Embeddings

Categorical modularity is a low-resource intrinsic metric for evaluation of word embeddings. We provide code for the community to:
- Generate embedding vectors and nearest-neighbor matrices of lists of core words
- Calculate the following scores for said lists of words:
  - General categorical modularity with respect to a fixed list of semantic categories
  - Single-category modularity with respect to a fixed list of semantic categories
  - Network modularity of emerging categories from the nearest-neighbor graph created by a community detection algorithm
- Over several language/model pairs, calculate the correlation between any of the above three modularity scores and performance scores on the following downstream tasks:
  - Sentiment analysis on IMDB movie reviews
  - Word similarity based on SEMEVAL 2017 word pairs
  - Bilingual lexicon induction (both English - target language and source language - English)
  
 
 ## Summary of Usage
This repository can be used to take a word embedding model and a list of words labeled with semantic categories, generate embeddings for those words, calculate the general, single-category, and network modularities of the model with respect to the words/categories, and calculate the correlations of those modularities with downstream tasks. Example workflow (for further instructions on how to use each file, refer to subsequent sections in this README as well as the `--help` messages and docstrings in each file described):
 1. Download and unzip a word embedding model binary. Examples include [these binaries](https://fasttext.cc/docs/en/pretrained-vectors.html) for FastText, [these binaries](https://github.com/facebookresearch/MUSE#download) for MUSE, and [these binaries](https://github.com/jvparidon/subs2vec#downloading-datasets) for subs2vec.
 2. Obtain a list of words in the language corresponding to the embedding model, formatted in a single-column headerless txt file. See `core/words/dutch.txt` for an example of such a file.
 3. Obtain a list of category labels corresponding to the word list (i.e. if label n = k in the category list, then word n in the word list belongs to category k), formatted in a single-column headerless csv file. See `core/words/categories_3.csv` for an example of such a file. 
 4. Run one of the `*vector_gen.py` files inside `core` to produce a txt file of vectors corresponding to each word in the word list. For FastText-compatible model binaries such as FastText and subs2vec, use `core/ft_vector_gen.py`. For MUSE-compatible binaries, use `core/muse_vector_gen.py`.
 5. Run a `*matrices.py` file from `core` to generate an n by n nearest-neighbor matrix for your list of n words. Use `core/ft_matrices.py` for FastText-compatible model binaries and `core/muse_matrices.py` for MUSE-compatible model binaries.
 6. To calculate the general categorical modularity for your given language/model, run `core/general_modularity.py` using your category labels and the matrix that was generated in your `*matrices.py` run. The results we obtained are in `core/results_general_modularity`, and the labeling scheme is `[level]_[k].csv`, with levels and k values as specified in Sections 4 and 5 of our paper, respectively.
 7. To calculate unsupervised network modularity, run `core/unsupervised_modularity.py`. The results we obtained are in `core/results_unsupervised_modularity`, labeled `[k].csv`, with k as specified in Section 5 of our paper.
 8. To calculate single-category modularities, run `single_category/single_category_modularity.py` using your category file and generated matrix file. The results we obtained are in `single_category/data`, labeled the same way the general modularity results are labeled.
 9. Run the sentiment analysis task:
   a. Download the English IMDB movie reviews from [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). 
   b. Turn the csv into a tsv by renaming it in Terminal or in Finder.
   c. Translate the data into your target language by running `task_movies/movie_data_gen.py`. This will produce a txt file with just the reviews.
   d. Generate embeddings for each review by passing the aforementioned txt file into one of the `*movie_gen.py` files in `task_movies`. Use `task_movies/ft_movie_gen.py` for FastText-compatible embeddings and `task_movies/muse_movie_gen.py` for MUSE-compatible embeddings.
   e. Run the analysis task by running `task_movies/movie_task.py`. This will output a txt file with the average accuracy and precision scores over your desired number of trials. Our accuracy and precision results are in `task_movies/results`.
 10. Run the word similarity task: 
   a. Several example data files are in `task_wordsim/data`. If you would like to freshly translate a data file, use `task_wordsim/wordsim_trans.py`. You can also download data from [here](https://alt.qcri.org/semeval2017/task2/index.php?id=data-and-tools).
   b. Run one of the `wordsim*data_gen.py` files in `task_wordsim` to generate embeddings for your word pairs.
   c. Run the similarity task with `task_wordsim/wordsim_task.py`. This will print your average MSE loss over your desired number of trials to the console. Our results are in `task_wordsim/results`.
 11. Run the bilingual lexicon induction task:
   a. Obtain data. Several sample files are in `task_bli/data`. The naming convention is `[2-letter source language code]-[2-letter target language code].0-5000.txt` for training data and `[2-letter source language code]-[2-letter target language code].5000-6500.txt` for testing data. If you use custom files, make sure they conform to the formatting specifications listed under the `--help` messages in the `task_bli/translation*data_gen.py` files. Note as well that our runner runs both the to-English and from-English tasks in one go, so make sure to have both sets of files ready.
   b. Generate word embeddings by running the appropriate `task_bli/translation*data_gen.py` file.
   c. Run the induction task by running `task_bli/translation_task.py`. This will write a file with both the to-English and from-English average cosine similarities between predicted translations and true translations. Our results are in `task_bli/results`.
 12. Run steps 1-11 for several languages/models, recording downstream task scores along the way. For each task, compile the performance scores into a single-column csv.
 13. To calculate the correlation between a set of general or unsupervised modularity scores and a set of downstream performance scores, run `core/correlation.py`. 
 14. To calculate correlations with single-category modularities, run `single_category/single_category_correlation.py`.
 
 ## Dependencies
 - [Python 3.6+](https://www.python.org/downloads/)
 - [scipy](https://www.scipy.org/)
 - [numpy](https://numpy.org/)
 - [nltk](https://www.nltk.org/)
 - [fasttext](https://fasttext.cc/)
 - [google_trans_new](https://pypi.org/project/google-trans-new/)
 - [scikit-learn](https://scikit-learn.org/stable/)
 - [networkx](https://networkx.org/)
 
 ## Calculating Modularity
Given a list of words and a `.bin` or `.vec` embedding model, you can calculate several modularity metrics using the files in the `core` directory. Brief descriptions of files and functionalities (further formatting specifications for files and parameters can be found by running the `--help` command on each file or by consulting the docstring at the top of each code file):
- `core/ft_vector_gen.py`: pass in a file containing a list of words and a file containing an embedding model binary and create a file with 300-dimensional embedding vectors of each of the input words. Use for FastText-compatible models (e.g. FastText, subs2vec). The lists of words and categories we used are labeled by language and level in `core/words`. Usage (with respect to topmost level directory of this repo): 
```
python3 core/ft_vector_gen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE
```
- `core/muse_vector_gen.py`: same functionality as `core/ft_vectorgen.py`, but used for MUSE-compatible models (a different implementation than the FastText library). Additionally asks for a language parameter specification for use in MUSE embedding generation in case a word requires stemming. This language parameter should be specified as the full lowercase name of the language (e.g. `english`, not `en`). Usage:
```
python3 core/muse_vector_gen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE --language LANGUAGE
```
- `core/ft_matrices.py`: given a list of n words and an embedding model binary, generates a file with an n x n matrix where row i, column j = k represents the fact that word j is the kth-nearest neighbor of word i in the given embedding space. Use this file for FastText-compatible models only. Usage:
```
python3 core/ft_matrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE
```
- `core/muse_matrices.py`: same functionality as `core/ft_matrices.py` but for MUSE-compatible model binaries. Usage: 
```
python3 core/muse_matrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE --language LANGUAGE
```
- `core/general_modularity.py`: calculate general categorical modularity given a list of categories and a matrix as generated by `core/ft_matrices.py` or `core/muse_matrices.py`, postprocessed to remove [] characters. Usage:
```
python3 core/general_modularity.py --categories_file CATEGORIES_FILE --matrix_file MATRIX_FILE
```
- `core/unsupervised_modularity.py`: calculate modularity of unsupervised clusters for all categories given a matrix as generated by `core/ft_matrices.py` or `core/muse_matrices.py`, postprocessed to remove [] characters. Usage:
```
python3 core/unsupervised_modularity.py --matrix_file MATRIX_FILE
```
- `core/correlation.py`: code that can be used to calculate the Spearman rank correlations of one set of modularity scores(`modularity_file`) with one set of task performance metrics (`downstream_file`). See `core/data` for default files - your inputs should conform to the formatting of these files. Usage:
```
python3 core/correlation.py --modularity_file MODULARITY_FILE --downstream_file DOWNSTREAM_FILE
```

## Single-Category Modularity Correlations
Our paper also explores an extension of categorical modularity to single-category modularity, which we test on each of the 59 categories listed in our paper. The `single_category` directory contains code that can be used to calculate these single-category modularities and their correlations with downstream task performance. Brief descriptions of files and functionalities:
- `single_category/single_category_modularity.py`: given a list of category labels for words and a square matrix of nearest-neighbor relationships among words, calculates single-category modularities for each category and prints results to console. Usage: 
```
python3 single_category/single_category_modularity.py --categories_file CATEGORIES_FILE --matrix_file MATRIX_FILE
```
- `single_category/single_category_correlation.py`: given a file with modularity scores for a set of categories and a file with performance metrics for a particular tasks, writes an output file with correlations between performance metrics and modularities with respect to each category. See `single_category/data/3_2.csv` for how the `modularity_file` should be formatted (we recommend compiling modularities from `single_category_modularity.py` into a spreadsheet as we did), and see `single_categories/movies_accuracy.csv` for how the `metrics_file` should be formatted. Usage:
```
python3 single_category/single_category_correlation.py --modularity_file MODULARITY_FILE --metrics_file METRICS_FILE --out_file OUT_FILE
```

## Running Downstream Tasks
Our paper presents moderate to strong correlations of categorical modularity with four downstream tasks. We provide code to reproduce these tasks in the `task_bli`, `task_wordsim`, and `task_movies` directories.
### Sentiment Analysis
The first task we run is sentiment analysis of [IMDB movie reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Files and functionalities:
- `task_movies/movie_data_gen.py`: given a file with data (raw text of movie reviews) and a target language, generates an equivalent dataset translated into the target language. The language name is the full English name of the language (e.g. `finnish`), while the language code is the 2-letter code (e.g. `fi`, full listing of codes [here](https://www.loc.gov/standards/iso639-2/php/code_list.php)). Usage:
```
python3 task_movies/movie_data_gen.py --data_file DATA_FILE --target_language_name TARGE_LANGUAGE_NAME --target_language_code TARGET_LANGUAGE_CODE
```
- `task_movies/ft_movie_gen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for FastText-compatible models. Usage:
```
python3 task_movies/ft_movie_gen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/muse_movie_gen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for MUSE-compatible models. Usage:
```
python3 task_movies/muse_movie_gen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/movie_task.py`: given data in the form of vectors (importantly, assuming the first half are positive and the second half are negative), runs the task of sentiment analysis and outputs mean accuracy and precision over 30 trials. Usage:
```
python3 task_movies/movie_task.py --data_file DATA_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Word Similarity
The second task we run is word similarity calculation on pairs of words given in [SEMEVAL 2017](https://alt.qcri.org/semeval2017/task2/index.php?id=data-and-tools). Files and functionalities (all within `task_wordsim` directory):
- `task_wordsim/wordsim_trans.py`: translates source language dataset into target language of choice. Usage:
```
python3 task_wordsim/wordsim_trans.py --word_file WORD_FILE --source_language --SOURCE_LANGUAGE --target_language TARGET_LANGUAGE
```
- `task_wordsim/wordsim_ft_data_gen.py`: given a list of word pairs and similarity scores, generates a list of 3-dimensional vectors (Euclidean, Manhattan, and cosine distance between the words) as input into the word similarity task. Use for FastText-compatible model binaries only. Usage:
```
python3 task_wordsim/wordsim_ft_data_gen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsim_muse_data_gen.py`: same functionality as `task_wordsim/wordsim_ft_datagen.py` but for MUSE-compatible models. Usage:
```
python3 task_wordsim/wordsim_muse_data_gen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsim_task.py`: runs the word similarity task given the data (3D vectors) file, the label file, and the model name. Outputs mean MSE loss over 30 trials. Usage:
```
python3 task_wordsim/wordsim_task.py --data_file DATA_FILE --label_file LABEL_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Bilingual Lexicon Induction
Lastly, we experiment on the cross-lingual tasks of bilingual lexicon induction both to and from English. Files and functionalities (all within `task_bli` directory):
- `task_bli/translation_ft_data_gen.py`: given word pair training/testing files in both directions and model binaries, generates 300-dimensional embeddings of all the words (8 files total - 4-4 train-test split, 4-4 from-to split, 4-4 English-non-English split). Use for FastText-compatible model binaries only. Usage:
```
python3 task_bli/translation_ft_data_gen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translation_muse_data_gen.py`: same functionality as `task_bli/translation_ft_data_gen.py` but for MUSE-compatible model binaries. Usage:
```
python3 task_bli/translation_muse_data_gen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translation_task.py`: given vectorized data files, runs the BLI task in both directions and outputs mean cosine similarity as a performance metric for both directions. Usage:
```
python3 task_bli/translation_task.py --target_train_to_file TARGET_TRAIN_TO_FILE --english_train_to_file ENGLISH_TRAIN_TO_FILE --target_train_from_file TARGET_TRAIN_FROM_FILE --english_train_from_file ENGLISH_TRAIN_FROM_FILE --target_test_to_file TARGET_TEST_TO_FILE --english_test_to_file ENGLISH_TEST_TO_FILE --target_test_from_file TARGET_TEST_FROM_FILE --english_test_from_file ENGLISH_TEST_FROM_FILE --train_size TRAIN_SIZE --test_size TEST_SIZE --language LANGUAGe --model_name MODEL_NAME
```

We also provide some data files that can be used to run each of these code files with its default parameters. To run the files involving model binaries with our defaults, download FastText models from [here](https://fasttext.cc/docs/en/pretrained-vectors.html) and MUSE models from [here](https://github.com/facebookresearch/MUSE#download). Additionally, our paper discusses experiments on subs2vec embeddings, which can be found [here](https://github.com/jvparidon/subs2vec) - we used the model binaries under OpenSubtitles. 
