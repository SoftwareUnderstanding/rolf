# Research sOftware cLassification Framework (ROLF)


## Introduction

The aim of this work is to present a flexible methodology to classify scientific software with similar functionality. We define software with similar functionality as those software repositories that belong to the same category based on their available documentation. When we talk about flexibility we mean that new categories can be easily included in our classification method, so the number of categories can be efficiently increased. This is needed because new categories can appear at any time and we do not want our software to get outdated over time.

Extracting the software category/categories helps group software projects with similar aims or scope. This can save developers time to look for similar repositories without the need of taking a deeper look into them.

## Usage

Possibly option:

```
usage: python src/main.py [-h] {collect-readmes,
                                 preprocess,
                                 train_test_split,
                                 merge-csv,
                                 train_models,
                                 predict,
                                 evaluate}

Perform all the methods of the program.

positional arguments:
    collect_readmes     Collect readme files, create dataset.
    preprocess          Preprocess given csv data file.
    train_test_split    Makes train test split on given csv file.
    merge_csv           Merge given csv files into one.
    train_models        Train the models.
    predict             Predict with the given models.
    evaluate            Evaluate the predictions.

optional arguments:
  -h, --help            show this help message and exit
```

### Collect readmes:

```
python3 src/main.py collect_readmes -h
usage: python src/main.py collect_readmes [-h]
    --input_mode {csvfile,url}
    --input INPUT [--category CATEGORY]
    [--awesome_list_mode | --no-awesome_list_mode]
    [--githublinks_file GITHUBLINKS_FILE]
    --readme_folder README_FOLDER
    --outfolder OUTFOLDER
   [--redownload | --no-redownload]
   [--input_delimiter INPUT_DELIMITER]

Collect readmes from collected urls from given file rows.

optional arguments:
  -h, --help            show this help message and exit
  --input_mode {csvfile,url}
                        Set input mode. The input can be given by a csvfile or an url in comand line.
  --input INPUT         Give the input.
  --category CATEGORY   Set category of input url. (Required if url input_mode is used)
  --awesome_list_mode, --no-awesome_list_mode
                        Set mode of links to awesome list. (default: False)
  --githublinks_file GITHUBLINKS_FILE
                        Give file to save collected githubs if awesome lists are given.
  --readme_folder README_FOLDER
                        Path to the folder where readme files will be saved per category.
  --outfolder OUTFOLDER
                        Path to the folder, where database per category will be saved.
  --redownload, --no-redownload
                        Redownload the readmes. (default: False)
  --input_delimiter INPUT_DELIMITER
                        Set delimiter of input csv file (default: ";").
```

### Preprocessing

```
python3 src/main.py preprocess -h
usage: python src/main.py preprocess [-h] --preprocess_file PREPROCESS_FILE

optional arguments:
  -h, --help            show this help message and exit
  --preprocess_file PREPROCESS_FILE
                        Name of .csv the file with the preprocessed data. The data will be saved with the same filename adding "_preprocessed" suffix.
```

### Train test split

```
python3 src/main.py train_test_split -h
usage: python src/main.py train_test_split [-h] --train_test_file TRAIN_TEST_FILE [--test-size TEST-SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --train_test_file TRAIN_TEST_FILE
                        Name of the file to split.
  --test-size TEST-SIZE
                        Size of the test set (default: 0.2).
```

### Merge CSVs

This is applicable if you want to use multiple datasets for training:

```
python3 src/main.py merge-csv -h usage: python src/main.py merge-csv [-h] --files FILES [FILES ...] --outfile OUTFILE

optional arguments:
  -h, --help            show this help message and exit
  --files FILES [FILES ...]
                        List of csv files to merge with the same header row and ";" delimiter.
  --outfile OUTFILE     Path to outfile csv file with the results.
```

### Train models

```
python3 src/main.py train_models -h
usage: python src/main.py train_models [-h]
    --train_set TRAIN_SET
    --results_file RESULTS_FILE
    --out_folder OUT_FOLDER
    [--evaluation_metric EVALUATION_METRIC]
    [--gridsearch {nogridsearch,bestmodel,bestsampler,bestvectorizer,all}]
    [--all_categories ALL-CATEGORIES [ALL_CATEGORIES ...] | --additional-categories ADDITIONAL-CATEGORIES [ADDITIONAL-CATEGORIES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --train_set TRAIN_SET
                        Name of the csv file containing train set.
  --results_file RESULTS_FILE
                        Path to the file where results will be saved.
  --out_folder OUT_FOLDER
                        Path to the folder where models will be saved.
  --evaluation-metric EVALUATION-METRIC
                        Name of the key for evaluation (default: "f1-score_overall").
  --all_categories ALL_CATEGORIES [ALL_CATEGORIES ...]
                        List of all categories used. Use only if you want not the basic categories. BASE-CATEGORIES=['Natural Language Processing', 'Computer
                        Vision', 'Sequential', 'Audio', 'Graphs', 'Reinforcement Learning']
  --additional_categories ADDITIONAL_CATEGORIES [ADDITIONAL_CATEGORIES ...]
                        List of categories adding to basic categories. BASE-CATEGORIES=['Natural Language Processing', 'Computer Vision', 'Sequential', 'Audio',
                        'Graphs', 'Reinforcement Learning']
```

### Predict

```
python3 src/main.py predict -h
usage: python src/main.py predict [-h] --inputfolder INPUTFOLDER --test-set TEST-SET --outfile OUTFILE

optional arguments:
  -h, --help            show this help message and exit
  --inputfolder INPUTFOLDER
                        Path of folder with the models.
  --test-set TEST-SET   Name of the csv file containing the test set.
  --outfile OUTFILE     Path to outfile csv file with the results.
```

### Evaluate

```
python3 src/main.py evaluate -h
usage: python src/main.py evaluate [-h]
    --inputfile INPUTFILE
    --outfile OUTFILE
    [--all-categories ALL-CATEGORIES [ALL-CATEGORIES ...] | --additional-categories
    ADDITIONAL-CATEGORIES [ADDITIONAL-CATEGORIES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --inputfile INPUTFILE
                        Path of the csv file with the predictions.
  --outfile OUTFILE     Path of the json file to write scores.
  --all-categories ALL-CATEGORIES [ALL-CATEGORIES ...]
                        List of all categories used. Use only if you want not the basic categories. BASE-CATEGORIES=['Natural Language Processing', 'Computer
                        Vision', 'Sequential', 'Audio', 'Graphs', 'Reinforcement Learning']
  --additional-categories ADDITIONAL-CATEGORIES [ADDITIONAL-CATEGORIES ...]
                        List of categories adding to basic categories. BASE-CATEGORIES=['Natural Language Processing', 'Computer Vision', 'Sequential', 'Audio',
                        'Graphs', 'Reinforcement Learning']
```


## Install poetry environment

After cloning the repository execute:

```
cd rolf
poetry install
```

(If poetry is not installed, execute: ```pip install poetry``` first)

Once the instalation is finishes, activate the environment using the command:

```
poetry shell
```

If this does not works, activate the environment manually:

```
source ~/.cache/pypoetry/virtualenvs/rolf-*/bin/activate
```
