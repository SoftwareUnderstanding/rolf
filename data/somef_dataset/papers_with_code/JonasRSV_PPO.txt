![Imgur](https://i.imgur.com/9v1BFS3.png)

### [Resources](#resources-1)
### [Requirements](#requirements-1)
### [Summaries](#summaries-1)
### [Demos](#demos-1)
  * [Continous Problem](#continous)
  * [Discrete Problem](#discrete)

### Resources 
  * https://arxiv.org/pdf/1707.06347.pdf (PPO)
  * https://arxiv.org/pdf/1506.02438.pdf (GAE)

### Requirements
  * python 3.x

Install
```bash
> pip3 install -r requirements.txt
```

Summaries
---
Start Tensorboard on the summaries directory, create one if there is none then run:
> tensorboard --logdir=summaries


<br>
<br>

## Demos
Solutions to two of the gyms from [openAI](https://gym.openai.com/)

## Continous

```bash
> python3 pendelum_demo.py -n -t (To train)
> python3 pendelum_demo.py -p    (To play trained model)
```

<a href="https://giphy.com/gifs/jxa5HFQeS3CLO2Sxdm"> <img width=351px src="https://media.giphy.com/media/jxa5HFQeS3CLO2Sxdm/giphy.gif" title="Pendelum demo"/></a>

![Imgur](https://i.imgur.com/vxiH7GY.png)


## Discrete

```bash
> python3 cartpole_demo.py -n -t (To train)
> python3 cartpole_demo -p       (To play trained model)
```

<a href="https://giphy.com/gifs/3rWc8qOYjVCfsgiqh2"> <img width=351px src="https://media.giphy.com/media/3rWc8qOYjVCfsgiqh2/giphy.gif" title="Cartpole demo"/></a>
