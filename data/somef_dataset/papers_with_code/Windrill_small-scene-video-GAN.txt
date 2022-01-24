# Scene GAN
Small scene generator via GAN

Repository includes the saved checkpoint files of the trained network and the model source code.

Examples
-------------------
<img src="./examples/1/1.jpg" width="80" height="45"/> <img src="https://github.com/Windrill/Scenery-Video-GANs/blob/master/examples/1/gout.gif" />

<img src="./examples/2/1.jpg" width="80" height="45"/> <img src="https://github.com/Windrill/Scenery-Video-GANs/blob/master/examples/2/gout.gif" />

<img src="./examples/3/1.jpg" width="80" height="45"/> <img src="https://github.com/Windrill/Scenery-Video-GANs/blob/master/examples/3/gout.gif" />

<img src="./examples/4/1.jpg" width="80" height="45"/> <img src="https://github.com/Windrill/Scenery-Video-GANs/blob/master/examples/4/gout.gif" />

<img src="./examples/5/1.jpg" width="80" height="45"/> <img src="https://github.com/Windrill/Scenery-Video-GANs/blob/master/examples/5/gout.gif" />

Network and Data
-------------------
Input image size: 80x45 pixels.
Output animation size: 80x45x32 pixels.

Training data retrieved from Flickr. This network is trained on approximately 7000 processed video clips each of the size 80x45x32 px. Training takes around half a day on a GeForce GTX1050.
Video data tagged with scenery related labels are selected for training data, such as 'sea'. The video dataset is not further processed.

Training
-------------------
The model is trained in two steps. A smaller model of size 40x23px is trained in the first. The source code provided on the repository directly reads from the saved checkpoint file of a trained 40x23px Neural Network model.

The model is a Convolutional Neural Network, and is also a Generative Adversarial Network.

Both the Wasserstein Loss function and the Mean Squared Error Loss function work well in this model. The model code in this repository uses the Mean Squared Error loss function.

Major references: "Progressive Growing of GANs" (https://arxiv.org/abs/1710.10196) incorporates two networks of different sizes; "Generating Videos with Scene Dynamics" (http://www.cs.columbia.edu/~vondrick/tinyvideo/paper.pdf) is the video GAN that inspired this project

Extensions
-------------------
The model can be modified trained in color easily with more training data. In such a situation the input image size would be 80x45x3, and the number of dimensions increase with the inclusion of an RGB channel. 7000 videos cannot accurately generate colored pixels in this model, and in most models.

The model's size can be easily extended under the condition that a GPU with a larger memory is available.
