## AutoEncoders 

- Self Supervised Algorithms(Unsupervised directed type of network)
# AE used for
	1) Feature detections
	2) Recomendation System
	3) Encoding Data
- Activation function(hyperbolic tangent)
- Softmax used in AE
# Training AE
	1) Start with array where lines(observations) corrosponds to users and the columns(featues) corrosponds to movies. Each cell(u,i)contains ratings from 1 to 5. 0 if no ratings of the movie by user.
	2) first user goes into the network. the input vector x={r1,r2,r3....rm} contains all its ratings for all the movies
	3) the input vector x is encoded into a vector z of lower dimension by a mapping function f(e.g. sigmoid)
		z=f(Wx+b) where W is vector of input weights and b the bais
	4) z is then decoded into the output vector of y same dimension as x. aimming to replicate the input vector x.
	5) the reconstuction erro d(x,y)=||x-y|| is computed. the goal is to minimize it.
	6) back Propogation frrom ritgh to lrft the error is backpropogated. The weights are updated according to how much they are responsible for th error. the learning rate decides by how much we update weights.
	7) Repeat steps 1 to 6 and update weights after each observation(Reinforcement Learning) or
	Repeat steps 1 to 6 but update weights only after a batch of observations(batch Learning).
# Regularization Techniques
	OverComplete Hidden Layers (hidden Layers > input Layers). 
	1) Sparse Auto Encoders
		- when Hidden Layers> input Layers. It can cheat if we use all hidden node.
		- To extract more features.
		- It enables not to use all the hidden layers at any time
	2) Denoising Auto Encoders
		- handling at input layers
	3) Contractive Auto Encoders
		- http://www.icml-2011.org/papers/455_icmlpaper.pdf
	4) Stacked Auto Encoder
		- http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
	5) Deep Auto Encoders
		- Stacked AE <> Deep Auto Encoders
		- RBMs are stacked on top of each other
		- https://www.cs.toronto.edu/~hinton/science.pdf
	
# Kaggle
- https://www.kaggle.com/shardaborse/recomendationsautoencoders
# Readings
- https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/
- https://blog.keras.io/building-autoencoders-in-keras.html
- http://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/
- https://www.cs.toronto.edu/~hinton/science.pdf
- http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
- http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf
- https://arxiv.org/pdf/1312.5663.pdf
- http://www.ericlwilkinson.com/blog/2014/11/19/deep-learning-sparse-autoencoders
