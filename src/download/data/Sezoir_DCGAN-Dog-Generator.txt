# DCGAN-Dog-Generator
This project uses the DCGAN method to generate pictures of dogs. 

### DCGAN
The DCGAN method trains 2 models simultaneously 
by a form of "rivarly" between them. The generator model will generate pictures, and will progressively
become better at generating dog images. The discriminator will judge these images as well as real
dog images and will learn to tell the difference between the real and fake images. The goal is
for the discriminator to no longer be able to tell the difference between real and fake.

More can be learned from: https://www.tensorflow.org/tutorials/generative/dcgan

### Credits
The dataset for this project was found at: 
https://www.kaggle.com/jessicali9530/stanford-dogs-dataset/data 

Credits to JadeBlue96 who has also done a DCGAN dog generator report found at:
https://github.com/JadeBlue96/DCGAN-Dog-Generator

Also the model was architecture and constants used from this paper:
https://arxiv.org/pdf/1511.06434.pdf
