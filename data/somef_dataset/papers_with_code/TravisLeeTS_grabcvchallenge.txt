# grabcvchallenge
Grab AI Challenge (Computer Vision)

Intro
Car's cplor, make and model recognition has been converted into a product and start selling by Sighthound. (it can even recognise the car plate.)
https://www.sighthound.com/products/cloud

I'm trying to create a similar classification model based on Efficientnet from Google AI Blog (Champion of recent ImageNet challenge) and intend to create a recognition API in future.
📚 Project page: https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html
📝 Paper: https://arxiv.org/abs/1905.11946
🔤 Code: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

The Problem
Given a dataset of distinct car images, can you automatically recognize the car model and make?

The Solution
THe code was run in Google Colab (TPU) mainly I'm too poor to afford a good GPU :)
Please run the module by following the sequence below: 
ultis.py -> preprocessing.py -> imagenet_input.py -> efficient_model.pu -> efficientnet_builder.py -> main.py -> eval_ckpy_main.py
