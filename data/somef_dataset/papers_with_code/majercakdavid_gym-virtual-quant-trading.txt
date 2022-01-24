# gym-virtual-quant-trading

The Gym Virtual Quant Trading demonstrates usage of Reinforcement Learning for trading. The main two parts are the customized OpenAI Gym environment (https://gym.openai.com/), used for trading simulation and agent(s) that interacts with the environment to achieve maximum gain by performing trading actions.

# Quick Start
To quickly get up and running the training script (DDPG + locally available csv file), do the following:

    1. Run this set of commands to create and active python virtual environment and afterwards install required dependencies
    ```{shell}
    > python -m venv ./env
    > ./env/Scripts/activate
    > pip install -r ./requirements.txt
    ```

    2. Install PyTorch accroding to: https://pytorch.org/get-started/locally/

    3. Run the following command:
    ```{shell}
    > python ./main.py
    ```

    4. View the progress of the model in Tensorboard:
    ```{shell}
    > tensorboard --logdir=./.cache/tensorboard
    ```

# Environment
The environment is built on top of the OpenAI Gym library (https://gym.openai.com/). The implementation of different environment can be found in `/gym_virtual_quant_trading/envs/` folder. There are two main environment for now:
    - `BaseTradingEnv`: an abstract environment that defines unified interface for all environments in the project
    - `PaperTradingEnv`: provides example value function implementation for agent evaluation and possibly light backtesting

# Agent
The main aim of the project is the flexibility when it comes to the agents. Everyone should be able to try out whatever SOTA agent. All of the agents are located in `agents` folder. Currently the project contains two agents:
    - `BaseAgent`: defines unified interface for all agents
    - `DDPG`: implements the Deep Deterministic Policy Gradient (Actor-Critic) approach (https://arxiv.org/abs/1509.02971)

# TODO
## Functionality:
    - Save model checkpoints during training
    - Penalize agent by every sell trade by number of days that are left up until the stock is held for at least year and so reward agent for every trade that it held for more than year by number of extra days
    - Penalize agent by tax depending on period for which the stock is held 
    - Implement data connector for Alpha Vantage

## Other:
    - Test functionality (unit/integration tests)
    - Split the available data so the model can be evaluated
    - Investigate possibilities of parallelization in training
    - Improve the README file
    - Move the TODO section of README file to the Issues in GitHub