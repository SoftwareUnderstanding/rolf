## SimCLR: Self-Supervised Learning approach to learn contrasting representation between images.

![SimCLR](https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif)

Why Self-Supervised Learning?

* Existing Deep Learning models requires large amount of annotated data to perform well.
* Annotating data is resource intensive task.
* Large amount of unlabeled data is easily available.
* Today, self-supervised models performs, if not better than supervised, at least same
as supervised learned models.

What is Self-Supervised Learning?

* Self-Supervised learning learns the representation from unlabel data in unsupervised fashion.
* Contrastive Learning is the widely used self-supervised learning method.
* Contrastive Learning:
    Contrastive methods aim to learn representations by enforcing similar elements to be equal 
    and dissimilar elements to be different. In recent months, we have seen an explosion of 
    unsupervised Deep Learning methods based on these principles. In fact, some self-supervised 
    contrastive-based representations already match supervised-based features in linear 
    classification benchmarks.

    The core of contrastive learning is the Noise Contrastive Estimator (NCE) loss.

* NCE: 
    Consider an input x and its variant x+, meaning x and x+ belong to similar class. Thus 
    x+ is a positive example of input x and similarly consider a example x- which is dissimilar to
    x, hence x- becomes the negative example for input x.

    Now, NCE tried bring the embeddings of input x and x+ closer together and embedding of x and x-
    farther apart.

To maximize the efficient representation of an input belonging to a particular class, the input 
should be contrasted against wide variant of negative examples.

A Simple Framework for Contrastive Learning of Visual Representations - SimCLR

SimCLR uses existing ResNet architectures as the backbone and then appends two fully connected layer
with relu activation function, generates a embedding of size 512, 1024 etc.

For a given input image, we create two correlated copies of the images by applying transformations to
the images like random crop, resize, gaussian blur, color distortion etc.

Downside of SimCLR is the number of parameters increases along with the time taken to run each experiment.

## To run SimCLR

```python
git clone https://github.com/Mayurji/SimCLR
pip install -r requirements.txt
python training.py --batch_size=32 --lr=0.003
```
