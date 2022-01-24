# CycleGAN

*I'm doing this for education and fun.*

### Use guide
I've taken some util and pipeline stuff from another repo (described later).  
I've seen some of these techniques before, so I thought it is the right place to taste them.

*Easy way:* download playground.ipynb and follow instructions.

*Hard way:*  
1. You **must** have train.py in the same dir (for example, *project/*) as model.py and datasets.py.
2. That *project/* dir **must** also include another dirs: *datasets/*, *saved_models/*, *generated_images*
3. In *datasets/* you should place your dataset directory *\*dataset_name\*/*, which **must** contain:
    * *\*dataset_name\*/testA/*
    * *\*dataset_name\*/testB/*
    * *\*dataset_name\*/trainA/*
    * *\*dataset_name\*/trainB/*
    * In these directories you should store your images.
5. In *saved_models/* and *generated_images/* you **must** create directories named again *\*dataset_name\*/*, where stuff from training will be saved.
6. Execute train.py. Arguments:
    * "--start_from", type=int, default=0, help="epoch number to start from; 0 for training from scratch"
    * "--num_epochs", type=int, default=200, help="number of epochs"
    * "--dataset_name", type=str, default="horse2zebra", help="name of the dataset"
    * "--batch_size", type=int, default=10, help="number of samples in batch"
    * "--img_height", type=int, default=256, help="image height in pixels"
    * "--img_width", type=int, default=256, help="image width in pixels"
    * "--channels", type=int, default=3, help="number of image channels"
    * "--num_residual", type=int, default=9, help="number of residual blocks"
    * "--lambda_cycle", type=float, default=10.0, help="cycle loss weight"
7. Model weights will be stored at *project/saved_models/\*dataset_name\*/*;  
   Generated images will be stored at *project/generated_images/\*dataset_name\*/*;

**My dataset can be found here: https://cutt.ly/wul8Xi5**

### Progress
For now I've implemented by myself CycleGAN architecture according to https://arxiv.org/pdf/1703.10593.pdf  

Specifically:  
* Generator model
* Discriminator model
* Training cycle

Also, I didn't want this homework to be another casual .ipynb project, so I decided to split it into modules.  
I've never did this before, deadline is near, but I need to learn this stuff for **Final Project \\[T]/**, so...   
This split was inspired by https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/cyclegan
("Inspired" quite weak word here. In fact I've just copied *most* of his pipeline.)  

Also by deadline reasons some util stuff was taken from that repo.  

Specifically:
* *sample_images* function to generate and store generated fake images in pretty neat way;
* entire logging pipeline; 
* *FakeImageBuffer* was also taken from there. However, maybe here credits should go to original CycleGAN repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
* nice stuff with *parser* which I've seen many times before. I'm glad that I've finally made it by myself (particularly).

All another stuff should be written by me if I'm not mistaken.
### Results
TL;DR: quite bad results to be honest.

I have some bad issue here: all my losses just stucked near some points. But I don't see where I've made a mistake.  
Models still learn some stuff, but it's not enough even for horse2zebra transmuting. So I haven't begun my own idea yet.  
AFAIK, my model should be identical to stuff, which were proposed in mentioned paper. Because they described their architecture quite clear=)  
But it's still not so powerful.
Anyway, I have one last chance (by time reasons) to reproduce horse2zebra once more.  
I've added identity loss (*removed already*), which was used in paper, but for another tasks (my images were tinted on generating too; I suppose it can help here too);  
I've added schedulers, because I forgot about decaying for the first time, lol.  
And my **hope**: I forgot about weight initialization as well. Now I've fixed it.  

*Upd.:* none of this ideas worked. But! I've rewatched some materials on segmentations and saw, that there is no normalization on output layers for Segnets/Unets.
AFAIK (thanks, Google) it's common practice to not include it (however, original paper on CycleGAN does), so I've deleted it.
Results are visibly better now. I don't have time for horse2zebra, so I'm training my... "Water to Wine" model:D
I hope, one shouldn't be The Divine One to have the ability to transform water into wine.

*Upd.2:* it really works! Not so good, because my dataset is too small, I guess. Also it is clear, that I should have added transformations, but I'll fix this later.  

Here are some results:  
![Entire](https://i.ibb.co/S7qybCN/entire.png)  
As you can see, it also transforms glass into wineglass. ~~So maybe it's more powerful than God himself.~~ Maybe it's the task where identity loss will work nice.
Also all problems caused by lack of transforms are seen here.  

![water2wine](https://i.ibb.co/K6vrXB4/water2wine.png)  ![wine2water](https://i.ibb.co/TtGH6Nm/wine2water.png)   
And here I think the most interesting result: it seems that model can distinguish water from glass and transform them separately: water and wine flows correctly transformed here.

### Problems
There are 2 huge problems:
1. Model wasn't supposed to transform glass into wineglass:D However, it was expected.
2. Model completely destroys image backgrounds.

I think both of these problems are caused by dataset. First, it is quite small for such a task. Second, it contains only images of water in glass and wine in glass. And this images pretty similar to each other. Even background: water often depicted with light background and wine often depicted with dark one.  
I think that here could help adding identity loss.

Not a huge problem, but I can't understand, why deleting InstanceNorm layer from output instantly forced model to work. Please, explain it for me if you read this:D
