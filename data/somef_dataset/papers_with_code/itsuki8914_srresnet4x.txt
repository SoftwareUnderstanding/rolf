# srresnet4x
Super resolution with srresnet using TensorFlow.
the attached model is supesialized in cartoons.

I referrd this paper:https://arxiv.org/abs/1609.04802

This implementation substitutes subpixel-convolution with deconvolution because building model time is very slow with subpixel-convolution.

this implementation is not GAN.

## Usage
put the images in the folder named "data". They are used for training. 
       
put the image in a folder named "val". They are used for validation.

when you set folders, training runs "python main.py". 

after training, test runs "python pred.py" It is executed on the images in the folder named "test". 


like this
```
main.py
pred.py
data
  ├ 000.png
  ├ aaa.png
  ...
  └ zzz.png
val
  ├ 111.png
  ├ bbb.png
  ...
  └ xxx.png
test
  ├ 222.png
  ├ ccc.png
  ...
  └ yyy.png
```


## example
left:nearest right:output

<img src = 'output/115_val.png' >


