# End-to-End Memory Network

![](/Notebooks/figs/E2EMN.png)

paper: [https://arxiv.org/abs/1503.08895](https://arxiv.org/abs/1503.08895)

## Getting Started

### Prerequisites

```
pytorch 0.4.1
argparse 1.1
numpy 1.14.3
matplotlib 2.2.2
```

### Notebooks

|Model|Link|
|:-|:-|
|Tutorial(not very nice yet)|[link](https://nbviewer.jupyter.org/github/simonjisu/E2EMN/blob/master/Notebooks/E2EMN_Tutorial.ipynb)|
|Test and Visualization Memories for each hop|[link](https://nbviewer.jupyter.org/github/simonjisu/E2EMN/blob/master/Notebooks/E2EMN_Test.ipynb)|

### Test Result

> Please Check for 4 models for 20 tasks in "[test_result.md](https://github.com/simonjisu/E2EMN/blob/master/test_result.md)"
> * pe: "position encoding"
> * te: "temporal encoding"

"adjacent" weight sharing method and encoding with "position encoding" helped a lot to improve tasks accuracy. Alse, "temporal encoding" for story helped.

## Blog

Link : [simonjisu.github.io](https://simonjisu.github.io/datascience/2017/08/04/E2EMN.html)

## Demo

Not ready yet
