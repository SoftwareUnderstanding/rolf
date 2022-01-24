Chandhini Grandhi, cgrandhi@ucsd.edu


## Abstract Proposal

This project builds an image to image translation using pix2pix model and combines it with style transfer to generate a stylised image from sketches. The project takes in a dataset of faces of people obtained from [CUHK dataset](http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html) . It consists of two phases: The first model is a Pix2pix Generative Adversarial networks that takes in the image, does the processing required and generates the photos from this. Essentially, this step involves translating edges to faces. The second model is the Neural Style transfer whose content image is the image generated from pix2pix model and style image is chosen by the user.The final generated image is the stylized version of face image generated from edges. 

I first built the models and experimented with the dataset. Then, I used some of the images drawn by my friends (Available in data/user-images) and generated standalone faces from user sketches and performed style transfer on them.


## Project Report

The report is available [here](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/tree/master/report)


## Model/Data

Briefly describe the files that are included with your repository:
- data : Input and generated output images of the pix2pix model are located in the data folder
- trained models: single_test contains the trained checkpoint for the pix2pix model
- pix2pixtensorflow : contains the cloned version of [pix2pix model](https://github.com/affinelayer/pix2pix-tensorflow)


## Code

Your code for generating your project:
- pix2pix model: Followed the steps in [pix2pix model](https://github.com/affinelayer/pix2pix-tensorflow)
- Style transfer model: style_transfer.ipynb

## Results

Two versions of results are shown below
1. Generated stylized images from the validation dataset during testing
- Input Image <br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/faces_test/images/f-021-01-inputs.png)

- edgestoface generated from pix2pix model <br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/faces_test/images/f-021-01-outputs.png)

- stylized image<br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/style_transfer_results/validation-data.png)

2. Generated stylized images from user inputs (my friends)
- Input Image <br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/data/user-images/resized/autodraw-3.png)

- edgestoface generated from pix2pix model<br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/data/user-images/output/output-autodraw3.png)

- stylized image<br />
![Alt Text](https://github.com/ucsd-ml-arts/ml-art-final-chandhini-g-1/blob/master/style_transfer_results/user-image.png)

## Technical Notes

- The code runs on Google CoLab
- The code requires pip, TensorFlow, OpenCv libraries to run.


## Reference

References to any papers, techniques, repositories you used:
- Papers
- https://arxiv.org/abs/1611.07004
- https://arxiv.org/abs/1508.06576
- Repositories 
- https://github.com/affinelayer/pix2pix-tensorflow
- Blog posts 
- https://ml4a.github.io/guides/Pix2Pix/
