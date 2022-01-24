# Description

AWS-based solution of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer) app.

<p float="left">
  <img src="./images/ex1.png" alt="Made with Angular" title="Angular" width="320"/>
  <img src="./images/ex2.png" alt="Made with Bootstrap" title="Bootstrap" width="320"/>
</p>

## How it works

NST is quite heavy on computation resources, thus GPU enabled instances of EC2 - P series - is used. Since running such instance on permanent basis is quite a luxury even in production, cost-effective AWS batchis selected to run style transfer on demand.

### AWS Labda based `python-telegram-bot` function

This function uploads received images to s3 bucket and handles s3 events on new images in s3 bucket - depending on the name of an image (content/generated) it invokes `AWS batch` (content) or sends generated image (generated).

### AWS Batch - Tensorflow docker image based on [AWS DL Container images](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-images.html)

Custom image runs the bootstrap script `entrypoint.py` which downloads tensorflow implementation of NST, fetches style and content images from s3 bucket, runs interference, uploads generated image and terminates container

### S3

CloudFormation template creates a temporary bucket with the stack which is used to store intermediate images and trigger notification event to drive Lambda function.

# How to build and deploy

## Prerequisites

* AWS account
* `awscli`
* `sam`

## Steps

1. Clone this repo and cd to it;
2. Build docker image and [push it to ECR](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-custom-images.html):
```
cd docker
docker build -t nst:latest .
```
3. Replace `JobDefinition/Properties/ContainerProperties/Image` in `cloudformation/template.yaml` with built image;
4. Paste your values into variables:

```
export BUCKET_NAME=name
export TELEGRAM_BOT_API_KEY=key
```

5. Build lambda package:
```
# sam build --use-container -t cloudformation\template.yaml -s .
```
6. Upload the package to s3:
```
# sam package --s3-bucket $BUCKET_NAME --output-template-file package.yaml
```
7. Deploy the stack to AWS:

```
# aws cloudformation deploy --template-file package.yaml --stack-name nst --capabilities CAPABILITY_IAM --force-upload --parameter-overrides "ApiKey=TELEGRAM_BOT_API_KEY" "BucketName=$BUCKET_NAME"
```

8. Download [VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) model to `$BUCKET_NAME` S3 bucket along with `style.jpg` image - this image style will be used in image processing;
9. Post an image to telegram bot and wait until app will process it.

# Reference

*deeplearning.ai for implementation of NST model.*

The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups from which we drew inspiration. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team.

* [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Harish Narayanan, Convolutional neural networks for artistic style transfer.](https://harishnarayanan.org/writing/artistic-style-transfer/)
* [Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style".](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
* [Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition] (https://arxiv.org/pdf/1409.1556.pdf)
* [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/)
