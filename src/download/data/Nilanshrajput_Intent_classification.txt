# Intent_classification

###### The project uses **Mlflow** for experiment tracking, and **Pytorch Lightning** and **hugging-face Transformers** Library
To start mlfow please check the documentaion here https://mlflow.org/docs for installation.
### Running the code
To run the example via MLflow, navigate to the `Intent_classification/src` directory and run the command

```
mlflow run .
```

This will run `model.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda

```

### Viewing results in the MLflow UI

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).

### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train model. Training can be interrupted early via Ctrl+C
2. gpus - Number of GPUs
3. accelerator - [Accelerator backend](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags) (e.g. "ddp" for the Distributed Data Parallel backend) to use for training. By default, no accelerator is used. 
4. batch_size - Input batch size for training
5. num_workers - Number of worker threads to load training data
6. lr - Learning rate

For example:
```
mlflow run Intent_classification -P max_epochs=5 -P gpus=1 -P batch_size=32 -P num_workers=2 -P learning_rate=0.01 -P accelerator="ddp"
```
Or to run the training script directly with custom parameters:

```
python Intent_classification/src/model.py \
    --max_epochs 5 \
    --gpus 1 \
    --accelerator "ddp" \
    --batch_size 64 \
    --num_workers 2 \
    --lr 0.001
```


## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).

## 1.Dataset

The data contains more than 2000 user queries that have been generated for each intent with crowdsourcing methods.

The Dataset is seggregated into train.csv, valid.csv and test.csv and available in the Dataset folder.

The dataset is categorized into seven intents such as:

1.  **AddToPlaylist**

2. **BookRestraunt**
 
3. **GetWeather**
 
4. **PlayMusic**
 
5. **RateBook**
 
6. **SearchCreativeWork**
 
7. **SearchScreeningEvent**


#### Link to the Github-repo: https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines

#### Link to the paper: https://arxiv.org/abs/1805.10190


 
## Model:

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

**Link to the BERT paper:** https://arxiv.org/abs/1810.04805


A blog explaining about transformers and evolution of BERT: https://jalammar.github.io/illustrated-bert/ 
