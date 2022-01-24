# DenseNet in Keras
A short, intelligible implementation of DenseNet in Keras <br>
paper link: www.arxiv.org/abs/1608.06993 <br>
### How to use?
In the DenseNet.py, I defined a function called "DenseNet" with one parameter(config), so this means if you want to use DenseNet, what you need to do is just provide corresponding config, the function DenseNet will return a Model, you can apply it in training. <br>
I prepare four configs for you, which are all in Config.py and can be found in paper, they are config121, config169, config201, config264. e.g. if you want to use DenseNet121, you may call DenseNet(config121). <br>

### Can I use my own config?
Yes! You can make you own config exactly, so the config is a python list with 8 items, each item is a dictionary: <br>
1st: a dictionary contains the image dimensions(image_size), number of classes(num_classes), and growth rate(k) <br>
2nd: single number stands for number of repeats(t) <br>
3rd: single number stands for compression(theta) <br>
(2) and (3) repeat for 2 time <br>
8th: same as 3rd
example: <br>
```
config = [
    {'input_size' : (224, 224, 3), 'num_classes' : 1000, 'k' : 32},
    6,
    0.5,
    12,
    0.5,
    32,
    0.5,
    32
]
```

If you have any questions, please leave a message in `Issue`, I will reply as fast as I can
