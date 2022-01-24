# NeuralNetAugmentations
Application of different augmentation techniques over a neural network

This assignment compares the behaviour of different augmentation techniquesover a resnet20


Dataset: CIFAR-10


CIFAR10 images are of size 32 × 32 and contains pictures corresponding
to 10 classes (e.g. airplane, cat, dog, ...) and our goal is deducing which class the image corresponds to.


Neural Network Configuration:
Optimizer: Adam,
Loss: Cross-entropy,
learning rate: 0.001,
Each image is normalized to have mean zero.


Following were achieved as a part of this project:


1. Train the Resnet model without augmentation and report the results.


2. Mixup augmentation is based on the paper https://arxiv.org/pdf/1710.09412.pdf. As
the name suggests, it mixes a pair of training examples (both inputs and labels). Given a pair of
training example (x1, y1), (x2, y2), we obtain the augmented training example (x, y) via
x = λx1 + (1 − λ)x2 y = λy1 + (1 − λ)y2
where mixing parameter λ has β distribution2 with parameter α.
TODO: Implement mixup and report the results for α = 0.2 and α = 0.4. Note that, in each minibatch,
all training examples should have mixup transformation before gradient calculation (e.g. from original
minibatch obtain a new minibatch by mixing random pairs of training examples).


3. Cutout augmentation is based on the paper https://arxiv.org/pdf/1708.04552.pdf. For
each training image with 50% probability, keep the image intact. With 50% probability, select a
random pixel which serves as the center of your cutout mask. Then, set the square mask of size K × K
pixels around this center pixel to be zero. Note that part of the mask is allowed to be outside of the
image. For visualization, see Figure 1 of the paper.
TODO: Implement and use cutout augmentation with K = 16 and report the results.



4. Standard augmentation applies horizontal flip and random shifts. See the website https://
machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
for illustrations. Given an input image, first you shift it left-right and up-down as follows. Pick two
independent integers k1, k2 uniformly between [−K, K] range. Move image upwards by k1 and rightwards by k2 pixels (negative value means downwards and leftwards). Zero pad the missing pixels. After
this random shift, with 50% probability, apply a horizontal flip on the image.
TODO: Implement standard augmentation with K = 4 and report the results3



5. Combine all augmentations together. First apply standard and cutout augmentations on the
training images and then apply mixup to blend them. For mixup, use the parameter α that has higher
test accuracy. Report the results. Does combining improve things further?




6. Comment on the role of data augmentation. How does it affect test accuracy, train accuracy
and the convergence of optimization? Is test accuracy higher? Does training loss converge faster?
