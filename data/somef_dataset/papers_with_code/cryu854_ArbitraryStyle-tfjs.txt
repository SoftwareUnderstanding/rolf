# Arbitrary Style Transfer in Tensorflow js

This is an implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf) on Tensorflow 2 and Tensorflow js. **Demo website** : https://cryu854.github.io/ArbitraryStyle-tfjs/

The model runs purely on your browser, so your data will not be leaked.
<div align='center'>
<img src = 'images/src/web_demo.jpg' height="480px">
</div>
<br>

The network architecture is conposed of an encoder, a decoder, and an AdaIN layer. The encoder is fixed to the first few layer (up to relu4_1) of a pre-trained [VGG-19](https://arxiv.org/pdf/1409.1556.pdf). The decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to reduce checkerboard effects. Set `REFLECT_PADDING=True` to use reflection padding in both encoder and decoder to avoid border artifacts, but the model will not be able to be deployed on the browser.


<div align='center'>
<img src = 'images/src/architecture.PNG' height="240px">
</div>

## Image Stylization :art:

<div align = 'center'>
<a href = 'images/src/example.PNG'><img src = 'images/src/example.PNG' height = '640px'></a>
</div>

### Stylize an image
Use `main.py` to stylize a content image to arbitrary style. 
Stylization takes 29ms per frame(256x256) on a GTX 1080ti.

Example usage:
```
python main.py inference --content ./path/to/content.jpg   \
                         --style ./path/to/style.jpg \
                         --alpha 1.0 \
                         --model ./path/to/pre-trainind_model
```

### Content-style trade-off
Use `--alpha` to adjust the stylization intensity. The value should between 0 and 1 (default).
<div align='center'>
<img src = 'images/src/interpolate.PNG' height="200px">
</div>

## Training
Use `main.py` to train a new style transfer network.
Training takes 2.5~3 hours on a GTX 1080ti.
**Before you run this, you should download [MSCOCO](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) and [WikiArt](https://www.kaggle.com/c/painter-by-numbers) dataset**. 

Example usage:

```
python main.py train --content ./path/to/MSCOCO_dataset   \
                     --style ./path/to/WikiArt_dataset \
                     --batch 8 \
                     --debug True \
                     --validate_content ./path/to/validate/content.jpg \
                     --validate_style ./path/to/validate/style.jpg
```
      
### Convert a pre-trained model to tensorflow-js model
Use tensorflow-js converter to generate a web friendly json model.
If you use **reflection padding** in encoder or decoder, the converter will not work properly because the current version of tensorflow-js does not support the mirrorpad operator.

Example usage:
```
tensorflowjs_converter --input_format=tf_saved_model --saved_model_tags=serve  models/model models/web_model
```

## Requirements
- TensorFlow >= 2.0
- Python 3.7.5, Pillow 7.0.0, Numpy 1.18
- If you want to convert a pre-trained model to tensorflow-js model:
  - Tensorflowjs >= 2.0

## Attributions/Thanks
- Some images/docs was borrowed from Xun Huang's [AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)
- Some tfjs code formatting was borrowed from tensorflow.js example [Mobilenet](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet)
