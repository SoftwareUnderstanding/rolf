# Ainize-run-PointRend example

[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=github.com/kmswlee/ainized-PointRend)

PointRend is image segmentation as rendering

The PointRend neural network module performs point-based segementation predictions at adaptively selected locations based on 
an iterative subvision algorithm.

PointRend achieves higher sharpness on tricky object boundaries such as fingers than Mask R-CNN, and can be added on both semantic and instance segmentation. 

This module show intermediate results.
So if you want to use Point Rend, apply Point Rend to your instance segmentation or semantic segmentation project.


Ainize is done in the following steps:
1. click 'default'.
2. click 'try it out' and first, input mask-image file and second, input original-image file.
3. click 'submit' button.


# How to use
this is dockerized, it can be run using docker commands.

## Docker build
```
docker build -t pointrend .
```

## Docker run
```
docker run -d --rm -p 80:80 pointrend
```
Now the server is available at http://localhost:80. 

## image examples

### input image
![mask](./tree_mask.jpg)
![img](./tree.jpg)

### intermediate result image
<img src="/output.jpg" width="500" />  

## References
* https://github.com/JamesQFreeman/PointRend
* https://arxiv.org/pdf/1912.08193.pdf
