# [Doodle Recognition][1] 

Web app classsificator based on the [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset).<br>
Has Node.js and Keras version of the MobileNetv2 underhood. Generates predicts across 340 categories.

![Doodle-GIF](https://github.com/Arcady1/Doodle-Recognition-Web/blob/master/img/GitHub-GIF/Animation.gif)

### How to use

To clone and run this application, you'll need Git and Node.js (which comes with npm) installed on your computer. From your command line:

```
# Clone this repository
$ git clone https://github.com/Arcady1/Doodle-Recognition-Web.git

# Go into the repository
$ cd Doodle-Recognition-Web

# Install dependencies
$ npm install

# Run the app
$ npm build
$ npm start
```

npm dependencies:
```
"browserify": "^16.5.2",
"@tensorflow/tfjs": "^2.0.1",
"express": "^4.17.1",
"mathjs": "^7.1.0"
```

### Model details

See more in [this](https://github.com/Arcady1/Doodle-Recognition-Web/blob/master/model/Train_MobileNetV2_Imagenet_weights.ipynb) Jupiter Notebook with MobileNetv2 training pipeline.<br>
Model type: MobileNetV2<br>
Weights initialization strategy: random noise<br>
Main hyperparameters:
* batch_size = 256
* alpha = 1
* input_size = (64, 64, 1)

The training took 6 hours on Tesla P100 (Google Collab).

The [model folder](https://github.com/Arcady1/Doodle-Recognition-Web/tree/master/model) also includes:
* A notebook with Imagent version of MobieNetv2 and input size (64, 64, 3)
* Keras models and weights converters from .h5 format to TensorFlow.js Layers format

### Acknowledgments
[MobileNetV2: Inverted Residuals and Linear Bottlenecks, arxiv article](https://arxiv.org/pdf/1801.04381.pdf) - Original article with the MobileNetv2 description<br>
[TensorFlow JS documentation](https://www.tensorflow.org/js/tutorials/conversion/import_keras) - This article describe how to convert pre-trained Keras model to TensoFlow JS model<br>
[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset) - Dataset

### You may also like...
* [Pomodoro Bot](https://github.com/Arcady1/Telegram-Pomodoro-Bot) - Telegram bot with the pomodoro timer

### License
MIT



















[1]: https://doodle-recognition-web.glitch.me/
