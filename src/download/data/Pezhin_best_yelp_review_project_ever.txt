# CS547 Final Project: LSTM, bi-LSTM and Attention model
Source code, project reports, execution instructions for our implementations of LSTM, bi-LSTM and Attention models for Yelp Review Polarity.

## Team Members
* Sahand Mozaffari (<sahandm2@illinois.edu>)
* Efthymios Tzinis (<etzinis2@illinois.edu>)
* Zhepei Wang (<zhepeiw2@illinois.edu>)
* Hao Wu (<haow11@illinois.edu>)
* Peilun Zhang (<peilunz2@illinois.edu>)

## Folder Structures and File Descriptions

```
├── LICENSE
├── README.md
├── code
│   ├── data_loader
│   │   ├── __init__.py
│   │   ├── datatool.py   # Wrapper class for the data set
│   │   └── utils.py      # Utility functions to convert data set to data loader
│   ├── main.py           # Main rountine for a single run of experiment
│   ├── models.py         # Wrapper class for all three different model structures
│   ├── parallel_experiment_runner.py # Helper file to paralleling experiments 
│   ├── tools
│   │   ├── __init__.py
│   │   ├── argtools.py  # Util functions to get and parse command line arguments
│   │   └── misc.py         
│   ├── vis_results.py   # File to generate visualizations 
│   └── run.sh           # Script to run parallelly  
└── requirements.txt
```

## Quick Execution Instructions

**Prerequisite**
* Python >= 3.6.9 

**Execution**
1. Install the required libaries.
```
pip install -r requirements.txt
```
2. Execute using the existing script
```
cd code && ./run.sh 
```
This is equivalent to the following script
```
python parallel_experiment_runner.py -cad 0 1 2 3 --epochs 15 -M BLSTM
```
More options can be found in the next section.

## Available parameter options 
| Flag            | Description                                              | Type  | Default Value                  | Allowable Values               |
|-----------------|----------------------------------------------------------|-------|--------------------------------|--------------------------------|
| lr              | Learning Rate                                            | float | 3e-4                           |                                |
| cad             | CUDA Available Devices                                   | str   | ['2']                          | ['0', '1', '2', '3']           |
| epochs          | Number of epochs for training                            | int   | 20                             |                                |
| bs              | Batch Size                                               | int   | 128                            |                                |
| num_workers     | Number of workers                                        | int   | 4                              |                                |
| preprocess_path | Path of preprocess data folder                           | str   | "../data/preprocess_data"      |                                |
| data_path       | Path of data folder                                      | str   | "../data"                      |                                |
| ckp             | Path of check point folder                               | str   | "../check_point"               |                                |
| out_dir         | Path of output folder                                    | str   | "../out"                       |                                |
| seed            | Seed for random number generator                         | int   | 1                              |                                |
| wandb_project   | Wandb project name                                       | str   | "cs547"                        |                                |
| wandb_entity    | Wandb entity name                                        | str   | "wandb entity"                 |                                |
| vocab_size      | Vocabulary size                                          | int   | [8000]                         |                                |
| M               | The type of model used for prediction                    | str   | ['LSTM', 'BLSTM', 'BLSTM-Att'] | ['LSTM', 'BLSTM', 'BLSTM-Att'] |
| L               | Number of hidden layers in the RNN                       | int   | [1, 2, 3]                      |                                |
| H               | Number of hidden units for each layer in selected model  | int   | [128, 256]                     |                                |
| D               | Dropout rate applied on all layers                       | float | [0.0, 0.3]                     |                                |
| E               | Size of the output of the embedding layer for each token | int   | [256, 512]                     |                                |