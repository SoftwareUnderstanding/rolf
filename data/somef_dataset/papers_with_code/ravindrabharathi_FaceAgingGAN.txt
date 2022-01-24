# FaceAgingGAN
Face Aging using CycleGANs

An attempt to map age progression in faces in images from the UTKFace dataset using CycleGANs. CycleGANs were introduced in this paper titled Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (https://arxiv.org/abs/1703.10593) where the authors presented an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples.

For the images of faces with various ages we will be using the UTKFace dataset wich has a cropped image set of only faces marked with age , gender , race , etc.

We will be using following two good references that use CycleGAN in order to build and train our models

https://github.com/sungnam0/Face-Aging-with-CycleGAN

https://machinelearningmastery.com/cyclegan-tutorial-with-keras/


Face Aging GAN with complete training output : https://github.com/ravindrabharathi/FaceAgingGAN/blob/master/Face_Aging_CycleGAN.ipynb

Face Aging GAN with training output removed (since the file with training output is huge and usually doesn't render well when viewed on Github) : https://github.com/ravindrabharathi/FaceAgingGAN/blob/master/Face_Aging_CycleGAN-without-training-output.ipynb
