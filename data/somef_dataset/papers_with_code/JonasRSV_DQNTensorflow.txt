## DQN implementation in tensorflow
based on
  * https://arxiv.org/pdf/1312.5602.pdf
  * https://arxiv.org/pdf/1509.06461.pdf
  * https://arxiv.org/abs/1607.06450

### Requirements
  * Python3 3.x


Installation
```bash
> pip3 install -r requirements.txt
```

```bash
> python3 cartpole_demo.py -n (Train Model)
> python3 cartpole_demo.py -p (Play Trained Model)
```

Summaries
---
Start Tensorboard on the summaries directory, create one if there is none then run:
> tensorboard --logdir=summaries

#### Demo
<a href="https://giphy.com/gifs/1jaMfIL5LHFAdrjM3h"> <img width=399px src="https://media.giphy.com/media/1jaMfIL5LHFAdrjM3h/giphy.gif" title="Cartpole demo"/></a>
