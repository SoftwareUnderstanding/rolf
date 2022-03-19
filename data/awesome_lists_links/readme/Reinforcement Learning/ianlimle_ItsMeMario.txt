## Prerequisites

1. Install [conda](https://www.anaconda.com/products/individual)
2. Install dependencies with `environment.yml`

   ```
   conda env create -f environment.yml
   ```

   Check the new environment _mario-env_ is [created successfully](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

3. Activate _mario-env_ environment

   ```
   conda activate mario-env
   ```

4. If your shell is not properly configured to use `conda activate`, it may be a source line in your bash scripts that has to explicitly reference your conda installation path. You can reference your conda installation path with the following:
   ```
   CONDA_PREFIX=$(conda info --base)
   source $CONDA_PREFIX/etc/profile.d/conda.sh
   conda activate mario-env
   ```

## Running the application

To start the **training** process for Mario,

```
python train.py
```

This starts the _double Q-learning_ and logs key training metrics to `checkpoints`. A copy of `MarioNet` and current exploration rate will be saved.

To **evaluate** a trained Mario,

```
python test.py \
--checkpoint [PATH/TO/YOUR/CHECKPOINT/FILE]
```

This visualizes Mario playing the game in a window. Performance metrics will be logged to a new folder under `checkpoints`. Change the `--checkpoint`, e.g. `checkpoints/2021-08-06T22-00-00/mario_net_1.chkpt` to check a specific timestamp.

## Project Structure
```
├── app
|   ├── /public
|   ├── /src
|   ├── .gitignore
│   ├── package-lock.json
│   └── package.json
├── agent.py
├── config.py
├── environment.yml
├── metrics.py
├── net.py
├── server.py
├── test.py
├── train.py
├── wrappers.py
├── .gitignore
└── README.md
```

**app**
Graphic User Interface (GUI) folder containing HTML, CSS and other assets.  

**agent.py**
Defines a `Mario` class that helps Mario collect experiences, make actions given observations and updates the action policy.

**config.py**
Define parameters for the training and evaluation of Mario.

**environment.yml**
Define packages/dependencies required for the environment to be installed and run.

**metrics.py**
Defines a `MetricLogger` class that helps track training/evaluation performance.

**net.py**
Define Q-value estimators with a Convolutional Neural Network based `MarioNet` class.

**server.py**
Flask server with endpoints for serving the GUI to run the training and evaluation scripts.

**test.py**
Evaluation script that loops between the environment and the trained Mario.

**train.py**
Training script that loops between the environment and Mario.

**wrappers.py**
Defines environment pre-processing logic, including observation resizing, rgb to grayscale, etc.

## Key Metrics

- Episode: current episode
- Step: total number of steps Mario played
- Epsilon: current exploration rate
- MeanReward: moving average of episode reward in past 100 episodes
- MeanLength: moving average of episode length in past 100 episodes
- MeanLoss: moving average of step loss in past 100 episodes
- MeanQValue: moving average of step Q value (predicted) in past 100 episodes

## Pre-trained

Checkpoints folder for a trained Mario agent: https://drive.google.com/drive/folders/11e0kPqND14o1LITcmo-3-iLtkOPJUxt6?usp=sharing

## Run GUI

Make sure you have NPM installed

1. Go into `app` folder
2. Run `npm install`
3. Run `npm start`
4. Open a separate terminal and perform `python server.py`
5. The GUI should be running at `localhost:3000` while the server which interacts with the AI scripts is `localhost:5001`

## Resources

Deep Reinforcement Learning with Double Q-learning, Hado V. Hasselt et al, NIPS 2015: https://arxiv.org/abs/1509.06461

OpenAI Spinning Up tutorial: https://spinningup.openai.com/en/latest/

Reinforcement Learning: An Introduction, Richard S. Sutton et al. https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

Deep Reinforcement Learning Doesn't Work Yet: https://www.alexirpan.com/2018/02/14/rl-hard.html
