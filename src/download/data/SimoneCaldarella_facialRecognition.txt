What is the project?

the project consists in using a convolutional network to realize a facial recognition. 
The frameworks used are Tensorflow [https://www.tensorflow.org] and openCv [https://opencv.org] and the language used is python(3.6.4).

This project is based on the study of the “transfer learning” and the possibility to use a neural network already trained with millions of images.
[https://www.tensorflow.org/tutorials/image_retraining].

More specifically I started with a script that use openCv face detection and save all the frames of a ~20 seconds video of somebody face. After doing it with all the people you want, it starts the second phase. 
Now you need to use a tensorflow script call “retrain.py” that download from tensorflow hub a complete neural network already trained with lots images. The default network is called “Inception v3” [https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/] [https://arxiv.org/pdf/1512.00567.pdf]. After selecting the folder with the images to use for training, you have to run the script and wait about 10 to 30 minutes. The training consist in the training only of the last layer of the network.

After training the network you can use the it. But how? 
I wrote a second script with which you can do inference. You only need to “feed” the script with an image with one or more people (among those present in the training images) and the script (with openCv and tensorflow) will provide to write under their faces the most probable name corresponding to the person.

The goal of the project should be to use those scripts to recognize people that passes somewhere (if they are already known) and keep trace about it.

Here some helpful links:

Tflearn models [https://github.com/tflearn/tflearn/tree/master/examples/images]                                          Tensorflow-hub pre-trained models [https://www.tensorflow.org/hub/modules/]
