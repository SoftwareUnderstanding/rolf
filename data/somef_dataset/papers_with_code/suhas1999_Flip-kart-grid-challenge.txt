# Flip-kart-grid-challenge
Main Approach:-
	With the independent feature being the image itself and the dependent feature being two set of coordinates, our approach revolves around building a standard Convolutional Neural Network. The architecture of the CNN used is ResNet50,Resnet18 (https://arxiv.org/abs/1512.03385). A custom head is added to the CNN to get the desired 4 numbers as output.

Some key techniques used to improve the model:-
	1.) Data Augmentation.(Changing the brightness, contrast, rotation, zooming, warping of the given images to artificialy generate more data.)
	2.) Differential Learning rates.(https://towardsdatascience.com/transfer-learning-using-differential-learning-rates-638455797f00)
	3.) One cycle fitting policy (https://arxiv.org/abs/1803.09820)
	4.) Cyclical Learning rates (https://arxiv.org/abs/1506.01186)
	5.) Stochastic Gradient Descent with Restarts.(The basic idea is to reset our learning rate after a certain number of iterations so that we can pop out of the             local minima if we appear to be stuck.)



Main libraries used:- 
	Fastai (https://github.com/fastai/fastai) built on top of PyTorch.
	Numpy
	Pandas
	OpenCV
