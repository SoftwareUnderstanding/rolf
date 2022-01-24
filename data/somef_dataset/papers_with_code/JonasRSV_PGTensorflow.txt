![Imgur](https://i.imgur.com/dzSr8cs.png)

### [Resources](#resources-1)
### [Requirements](#requirements-1)
### [Summaries](#summaries-1)
### [Demo](#demo-1)


### Resources 
  * https://arxiv.org/abs/1509.02971 (DDPG Paper)
  * https://arxiv.org/abs/1607.06450 (Layer Normalization)
  * https://arxiv.org/pdf/1706.01905.pdf (Todo, parameter noise good with layer norm)
  * https://arxiv.org/pdf/1702.00032.pdf (Noise process used for exploration)


#### Requirements
  * python 3.x

Install
```bash
> pip3 install -r requirements.txt
```

Summaries
---
Start Tensorboard on the summaries directory, create one if there is none then run:
> tensorboard --logdir=summaries


#### Demo
```bash
> python3 pendelum_demo.py -n (To Train)
> python3 pendelum_demo -p    (To play)
```

<a href="https://giphy.com/gifs/pPhC9JJAvIOK3gfnz9"> <img width=399px src="https://media.giphy.com/media/pPhC9JJAvIOK3gfnz9/giphy.gif" title="Pendelum demo"/></a>
