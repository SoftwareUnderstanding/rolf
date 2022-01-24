# Colorizer Bot

This Deep Learning Project can colorize black & white images with Python.
This project in based [Richzhang paper](https://richzhang.github.io/colorization/). I intend to use this algorithm as a twitter bot


## Download the caffemodel and  prototxt 
    Models
    $ wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ./core/models/model.caffemodel
    
    Prototxt
    $ wget https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt -O ./core/models/deploy.prototxt



## Running

    Install dependecies
    $ pip install -r requirements.txt
    $ python app.py

## Results

<img alt="before" src="sample_bw.jpg" height=400 align="left">
<img alt="after" src="sample_predict.jpg" height=400 align="left">

<br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br><br></br>

## How does it work ?
![Colorful Image Colorization](https://richzhang.github.io/colorization/resources/images/net_diagram.jpg)


## TODO

* Implement as TwitterBot
* Better Engine structure
* Tests


---

#### References
- https://richzhang.github.io/colorization/
- https://arxiv.org/abs/1603.08511
- https://developer.twitter.com/en/docs
