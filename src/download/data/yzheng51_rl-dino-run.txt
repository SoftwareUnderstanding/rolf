# rl-dino-run

This project aims to

- Creates an agent to play T-rex Runner
- Compares the performance of different algorithms
- Investigate the effect of batch normalization

## T-rex Runner

This game environment is based on [this repo](https://github.com/elvisyjlin/gym-chrome-dino) and modify the reward function and preprocessing steps. To simplify the decision, there are only two actions in action space

- Jump
- Do nothing

![alt text](./images/dino-run.png "T-rex Runner")

## Hyperparameter Tuning

There are many hyperparameters in Reinforcement Learning. In this project, we assume each hyperparameter is independent from others so that we can tune them one by one. Below table shows all parameters after tune.

| Hyperparameter  | Value     |
| --------------- | --------- |
| Memory Size     | 3 × 10^5  |
| Batch Size      | 128       |
| Gamma           | 0.99      |
| Initial epsilon | 1 × 10^−1 |
| Final epsilon   | 1 × 10^−4 |
| Explore steps   | 1 × 10^5  |
| Learning Rate   | 2 × 10^−5 |

## Training Results

### Comparison of different DQN algorithms

Using tuned hyperparameters run 200 epochs. Prioritized Experience Replay shows pretty bad effect because of weight update which is very time consuming and the game will keep runing when updating weight

![alt text](./images/exp-train.png "T-rex Runner")

### Comparison between DQN and DQN with Batch Normalization

![alt text](./images/exp-train-bn.png "T-rex Runner")

### Statistical Results in Training

| Algorithm         | Mean    | Std     | Max   | 25%    | 50%   | 75%     | Time (h) |
| ----------------- | ------- | ------- | ----- | ------ | ----- | ------- | -------- |
| DQN               | 537.50  | 393.61  | 1915  | 195.75 | 481   | 820     | 25.87    |
| Double DQN        | 443.31  | 394.01  | 2366  | 97.75  | 337   | 662.25  | 21.36    |
| Dueling DQN       | 839.04  | 1521.40 | 25706 | 155    | 457   | 956.5   | 35.78    |
| DQN with PER      | 43.50   | 2.791   | 71    | 43     | 43    | 43      | 3.31     |
| DQN (BN)          | 777.54  | 917.26  | 8978  | 97.75  | 462.5 | 1139.25 | 32.59    |
| Double DQN (BN)   | 696.43  | 758.81  | 5521  | 79     | 430.5 | 1104.25 | 29.40    |
| Dueling DQN (BN)  | 1050.26 | 1477.00 | 14154 | 84     | 541.5 | 1520    | 40.12    |
| DQN with PER (BN) | 46.14   | 7.54    | 98    | 43     | 43    | 43      | 3.44     |

## Testing Results

In testing stage, each algorithm uses the latest model and run 30 times

### Boxplot of all cases

![alt text](./images/exp-test.png "T-rex Runner")

### Statistical Results in Testing

| Algorithm         | Mean       | Std        | Min     | Max      | 25%     | 50%       | 75%        |
| ----------------- | ---------- | ---------- | ------- | -------- | ------- | --------- | ---------- |
| **Human**         | **1121.9** | **499.91** | **268** | **2384** | **758** | **992.5** | **1508.5** |
| DQN               | 1161.30    | 814.36     | 45      | 3142     | 321.5   | 1277      | 1729.5     |
| Double DQN        | 340.93     | 251.40     | 43      | 942      | 178.75  | 259.5     | 400.75     |
| Dueling DQN       | 2383.03    | 2703.64    | 44      | 8943     | 534.75  | 1499.5    | 2961       |
| DQN with PER      | 43.30      | 1.64       | 43      | 52       | 43      | 43        | 43         |
| DQN (BN)          | 2119.47    | 1595.49    | 44      | 5823     | 1218.75 | 1909.5    | 2979.75    |
| Double DQN (BN)   | 382.17     | 188.74     | 43      | 738      | 283.75  | 356       | 525.5      |
| Dueling DQN (BN)  | 2083.37    | 1441.50    | 213     | 5389     | 1142.5  | 1912.5    | 2659.75    |
| DQN with PER (BN) | 45.43      | 7.384      | 43      | 78       | 43      | 43        | 43         |

## Usage

Download the chrome driver from [this link](https://chromedriver.chromium.org) corresponding to your Chrome version and put the executable in the root path

Install all required modules by

```sh
pip install -r requirements.txt
```

Run the sample code

```sh
python main.py
```

## References

- DQN: [[paper]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)[[code]](./agent.py#L19-L136)
- Double DQN: [[paper]](https://arxiv.org/pdf/1509.06461.pdf)[[code]](./agent.py#L150-L168)
- Dueling DQN: [[paper]](https://arxiv.org/pdf/1511.06581/pdf)[[code]](./agent.py#L139-L147)
- DQN with Prioritized Experience Replay: [[paper]](https://arxiv.org/pdf/1511.05952.pdf)[[code]](./agent.py#L171-L209)
- Batch Normalization: [[paper]](https://arxiv.org/pdf/1502.03167.pdf)
