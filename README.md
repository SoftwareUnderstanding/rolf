# Research sOftware cLassification Framework (ROLF) [![DOI](https://zenodo.org/badge/413199643.svg)](https://zenodo.org/badge/latestdoi/413199643)

## Introduction

Software is increasingly important for research based on computational experiments. Software that is developed for this purpose is called research software. Research software can be simple such as visualization but it can be complex too like computational pipelines.

Developers and researchers often use tools developed by others. In order to find these tools, the developers need to search for them over the internet. The number of available software has a negative effect on the time spent searching. Researchers use 3 main approaches for finding the needed tool: general-purpose search engines, recommendations from a colleague, and scientific literature.

Suppose that we are looking for a software package that is able to perform Fourier transformation on audio. We would like to know some specific information about that package, for example, does it works on audio or other data, what input format can it process, what algorithm it uses, how can it be used, what is the output, etc. There are many questions that could arise, and to answer these questions a significant amount of developer time is required. There are tools that are able to provide this type of information about software, however, developing a machine-readable categorization is not trivial.

Scientific software can be classified based on its field and purpose. The class or category of software is also considered metadata. The classification is rather complex because there are no exact rules for defining the field of research/software. One software can belong to several categories or none of the known categories. To make sure that the used software samples are well-classified we have to rely on external vocabulary or other resources from the scientific community. 

The aim of this work is to present a flexible methodology to classify scientific software with similar functionality. We define software with similar functionality as those software repositories that belong to the same category based on their available documentation. When we talk about flexibility we mean that new categories can be easily included in our classification method, so the number of categories can be efficiently increased. This is needed because new categories can appear at any time and we do not want our software to get outdated over time.

Extracting the software category/categories helps group software projects with similar aims or scope. This can save developers time to look for similar repositories without the need of taking a deeper look into them.

## Usage

Possibly option: 

```
usage: python src/main.py [-h] {collect-readmes,
                                 preprocess,
                                 train-test-split,
                                 merge-csv,
                                 train-models,
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
python3 src/main.py train-test-split -h
usage: python src/main.py train-test-split [-h] --train-test-file TRAIN-TEST-FILE [--test-size TEST-SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --train-test-file TRAIN-TEST-FILE
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
python3 src/main.py train-models -h
usage: python src/main.py train-models [-h] 
    --train-set TRAIN-SET
    --test-set TEST-SET
    --results-file RESULTS-FILE
    --out-folder OUT-FOLDER
    [--evaluation-metric EVALUATION-METRIC]
    [--gridsearch {nogridsearch,bestmodel,bestsampler,bestvectorizer,all}]
    [--all-categories ALL-CATEGORIES [ALL-CATEGORIES ...] | --additional-categories ADDITIONAL-CATEGORIES [ADDITIONAL-CATEGORIES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --train-set TRAIN-SET
                        Name of the csv file containing train set.
  --test-set TEST_SET   Name of the csv file containing test set.
  --results-file RESULTS-FILE
                        Path to the file where results will be saved.
  --out-folder OUT-FOLDER
                        Path to the folder where models will be saved.
  --evaluation-metric EVALUATION-METRIC
                        Name of the key for evaluation (default: "f1-score_overall").
  --all-categories ALL-CATEGORIES [ALL-CATEGORIES ...]
                        List of all categories used. Use only if you want not the basic categories. BASE-CATEGORIES=['Natural Language Processing', 'Computer
                        Vision', 'Sequential', 'Audio', 'Graphs', 'Reinforcement Learning']
  --additional-categories ADDITIONAL-CATEGORIES [ADDITIONAL-CATEGORIES ...]
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

For any questions please contact: jenifer.girl.98@gmail.com
