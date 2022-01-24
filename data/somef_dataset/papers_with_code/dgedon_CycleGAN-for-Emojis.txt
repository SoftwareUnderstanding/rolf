# CycleGAN for Emoji Style Transfer

This is the repository for the project of module 3 in the [WASP](https://wasp-sweden.org/graduate-school/) 
graduate school course [Deep Learning and GANs](https://wasp-sweden.org/graduate-school/ai-graduate-school-courses/) by 
[Sofia Ek](http://www.it.uu.se/katalog/sofli286), [Carmen Lee](https://www.it.uu.se/katalog/carle978) and [Daniel Gedon](https://www.it.uu.se/katalog/dange246).

We are reimplementing the CycleGAN paper by Zhu et al, see [here](https://junyanz.github.io/CycleGAN/). We are taking the 
same setup as in assignment 4 of the course by Roger Grosse, see [here](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/).
That means that we take the accompanying dataset, containing emojis from Apple and Windwos and try to transfer the style between them.
The learnt transfer can be seen in the image below.

## Results

The following restults are obtained after training for 220 epochs from scratch.

![](doc/epoch220-app.png?raw=true)  
*Transfer from Apple style emojis to Windows style ones*

![](doc/epoch220-win.png?raw=true "Test")  
*Transfer from Windows style emojis to Windows style ones*

## Code Usage

To run the code with standard settings use
```bash
python main.py
```
The results above are obtained with
```bash
python --batch_size=16 --max_epochs=500 --learning_rate=3e-4 --beta1=0.5 --beta2=0.999 --loss_lambda=2
```