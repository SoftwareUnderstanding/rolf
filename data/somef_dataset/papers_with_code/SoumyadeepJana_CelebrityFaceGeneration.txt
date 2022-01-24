# CelebrityFaceGeneration

The aim of this project was to develop a Generative Adverserial Network that could generate completely new faces from random noise by completely learning the underlying probability distribution of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.The CelebA dataset contains over 200,000 celebrity images with annotations.The system can learn and separate different aspects of an image unsupervised; and enables intuitive, scale-specific control of the synthesis.The network architecture followed the standard DCGAN [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) architecture with slight modifications done.As you can see,the model performs quite well with the generated faces well formed and distinguishable.

### Preview of the Real Images
![Training Image](https://github.com/SoumyadeepJana/CelebrityFaceGeneration/blob/master/real.png)

### Preview of the Generated Images
![Training Image](https://github.com/SoumyadeepJana/CelebrityFaceGeneration/blob/master/generated.png)

### Inference of the Generated Samples
*The generated samples are largely white faces due to tha lack in diversity of the images belonging to different ethnicities.<br>
*The model was quite small with a dept of 512,thus detailed features of eyes,ears etc couldnt be modelled acccurately.<br>
*The number of epochs,the dept of the layers should be increased to achieve better results.
