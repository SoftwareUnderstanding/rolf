# The CycleGAN Notebook
In this project, I focus on the implementation of a cycle generative adversarial network model (CycleGAN) in an interactive way by using a jupyter notebook which is helpable to read and run for the training/inference of the model.
<br/><br/>

## CycleGAN Architecture
#### High Level
|[![](static/depict/high_level_from_A.png)]( https://hardikbansal.github.io/CycleGANBlog/ "Image from https://hardikbansal.github.io/CycleGANBlog/")| [![](static/depict/high_level_from_B.png)]( https://hardikbansal.github.io/CycleGANBlog/ "Image from https://hardikbansal.github.io/CycleGANBlog/")|
|:---:|:---:|
#### Low Level
|[![](static/depict/cyclegan_generator.png)](static/depict/cyclegan_generator.png "cyclegan_generator")| [![](static/depict/cyclegan_discriminator.png)](static/depict/cyclegan_discriminator.png "cyclegan_discriminator")|
|:---:|:---:|
<div></div><br/>

## How It Works
The CycleGAN model takes a real image from domain A and converts that image to a fake image in domain B. At the same time, it takes a real image from domain B and then converts it to a fake image in domain A. Here are some results that I ran on the horse2zebra dataset. The first row contains real horse images (domain A). The second row contains fake zebra images(domain B). The third row contains real zebra images (domain B). The last row contains fake horse images(domain A).

|||
|:---:|:---:|
|[![epoch 1](static/depict/epoch_1.png)](static/depict/epoch_1.png "epoch 1") epoch 1 | [![epoch 33](static/depict/epoch_33.png)](static/depict/epoch_33.png "epoch 33") epoch 33|
| [![epoch 66](static/depict/epoch_66.png)](static/depict/epoch_66.png "epoch 66") epoch 66 | [![epoch 99](static/depict/epoch_99.png)](static/depict/epoch_99.png "epoch 99") epoch 99 |
<div></div><br/>

## Technologies
- Python
- Pytorch
- Jupyter notebook
- Pillow
- Matplotlib
<br/><br/>

## Installation and Running
    $ git clone https://github.com/nhannguyencsd/vision_cyclegan.git
    $ cd vision_cyclegan
    $ python3 -m venv venv 
    $ source venv/bin/activate
    $ pip install -r static/libraries/requirements.txt
    $ jupyter notebook
* Once your jupyter notebook is opened, you can run a training_cyclegan.ipynb or inference_cyclegan.ipynb.</li>
* If you are not able to install libraries from requirements.txt or run on any notebooks, you are welcome [run](https://nhancs.com/project/2) the model on my website.
<br/><br/>

## Contributing
If you found any problems with this project, please let me know by opening an issue. Thanks in advance!
<br/><br/>

## License
This project is licensed under the MIT [License](LICENSE)
<br/><br/>

## References
The CycleGAN paper: [https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593) <br/>
CycleGAN datasets: [https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) <br/>
CNN Padding formular: [https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L13_intro-cnn-part2_slides.pdf](https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L13_intro-cnn-part2_slides.pdf) <br/>
Model architecture 1: [https://hardikbansal.github.io/CycleGANBlog/](https://hardikbansal.github.io/CycleGANBlog/) <br/>
Model architecture 2: [https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d3](https://towardsdatascience.com/cyclegan-learning-to-translate-images-without-paired-training-data-5b4e93862c8d) <br/>

