# Seq2seq model for Meme Generator

## Background
Meme, joined the meaning of vision and sentence, aims to make people feel hilarious for specific subject or culture. To create a meme is not easy because you need the comprehension and humor for subjects. Hence, I create the meme generator with the help of artificial intellegence though.

## Example
![](https://i.imgur.com/dQVYEQL.jpg)

## Model
Requirements
- nltk==3.5
- textblob==0.15.3
- gensim==3.8.3
- textaugment==1.3.4
- torch==1.7.1
- torchvision==0.8.2
```
pip3 install transformers
```
Preprocessing language data
```
python3 preprocess.py
```

Traning the model
```
python3 main.py --batch_size 64 --input_max_length 11 --output_max_length 14
```

## Experiment
> Pick an image then write something to generate the corresponding meme.

Loading the pretrained model you need to download the file https://drive.google.com/file/d/1m-vE0hVeTUzQ34y_OxA4UmD3laYsehSx/view?usp=sharing and put it to checkpoint directory
```
cp model.t7 Seq2seq-model-for-Meme-Generator/checkpoint/model.t7
```

Generate meme from random template
```
python3 demo.py 
```
Apply on your own image
```
python3 demo.py --image_path 'photo.jpg' --image_sent 'Hello world'
```

## References
- https://arxiv.org/abs/1409.3215
- https://github.com/huggingface/transformers
- https://arxiv.org/abs/1506.01497
- https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html
