# Defect Classification
*This is a classification model for classifying defects in mechanical parts.*

Here a pretrained model of  MobileNetV2 is used which is an open-source architechture in AI library of TensorFlow.
By taking the weights from that model and on top of it building one custom fully connected layer and finally the classification sigmoid layer to get the output according to the classes of our data.

MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks which obtains very good results on a wide array of Computer Vison tasks. Academic paper which describes MobileNets in detail and provides full results on a number of tasks can be found here: https://arxiv.org/abs/1704.04861.


## To run this clone the repository
```sh
git clone https://gitlab.com/pradyu1/queansbert.git
```

## Development environment

Before you start, make sure you have python3.6 installed and, if not follow:

```sh
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
```

After that create virtual environment with *Python3.6*
```sh
virtualenv --python=python3.6 myvenv
```

Then Activate virtual environment 
```sh
source ./myvenv/bin/activate
```

## Setup

Install other requirements for setup
```sh
pip install -r requirements.txt
```

## model_convert_test_tf114.tflite
This is the model which after training on our data is converted after quantization to tensorflow lite format to reduce the model size and be able to deploy in on mobile apps. 

## pipeline
This is the directory which contains the app to be deployed as server.

## Deploy the flask api
The model is wrapped in the flask application making it ready to deploy, just run this command to start it.
```sh
python3 -m pipeline.app
```


## Get Inference
To get inference from the model, in the "request.py" just input the image path for which output is required in the data
dictionary then run.
```sh
python3 request.py
```

