# Machine Learning Project
This project can detect the vehicle type in an image and circle out the object position.
<img src="https://github.com/ethanhou99/ml_learn/blob/master/output_images/test2.jpg" />
<img src="https://github.com/ethanhou99/ml_learn/blob/master/output_images/test7.jpg" />

## Instruction
To use this API, please set up the enviroment as follow:
1. Create a new folder and run terminal under this folder's path
2. Type in the folloing code:
   ```
   git init
   git clone https://github.com/ethanhou99/ml_learn
   ```
3. Save the images you plan to test to the 'test_images'(must be .jpg files)
4. Since the model is too large, you need to download the model from: 
   https://drive.google.com/open?id=1U3SxjOcr4FsYsG2gd9nVJHsJ-w4sJR18
   Then, save it to the main folder you created.
5. Right now, your main folder should have four sub-folders: font, output_images, resource, test_images;
   and main.py, SSD300.hdf5(model)
6. Run the mian.py
7. The recognize result are saved in the output_images folder

## Tagging method
1. The tagging method is from https://www.neurala.com/
2. Another tagging method is using labelImage.
3. Both of these two methods are shown as below:
<img src="https://github.com/ethanhou99/ml_learn/blob/master/images/tagging%20example.png" />
<img src="https://github.com/ethanhou99/ml_learn/blob/master/images/tagging%20exampleII.png" />
4. About 500 images are tagged to train the models.

## Training method
Two method are used to train the model:
   - Method1:https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku
   - Method2:https://github.com/ethanhou99/ml_learn/blob/master/model2.ipynb (also in the main folder model2.ipynb)
   - Again, model1 can be dowloaded from https://drive.google.com/file/d/1U3SxjOcr4FsYsG2gd9nVJHsJ-w4sJR18/view and model2 is in the main folder directly called model2.h5

## Comparision
The traing method is SSD, thus I'll compare SSD with another project's method - Unet

### SSD
- SSD is the first deep network based object detector that does not resample pixels or features for bounding box hypotheses and is as accurate as approaches that do. 
- Advantages
1. Features are pooled from different scales of the feature maps, and as a result the overall algorithm can detect objects at different scales and of different sizes and is more accurate than faster-RCNN;
2. As all the predictions are made in a single pass, the SSD is significantly faster than faster-RCNN.
- Disadvantages
1. Information flows unidirectional and the classifier network cannot utilize other directional information;
2. Too slow for real-time applications.
### Unet
- Unet network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.
- Advantages
1. Unet is really versatile and can be used for any reasonable image masking task;
2. High accuracy given proper training, adequate dataset and training time;
3. This architecture is input image size agnostic since it does not contain fully connected layers (!);
4. This also leads to smaller model weight size (for 512x512 U-NET - ca. 89mb);
5. Can be easily scaled to have multiple classes;
6. Code samples are abundant (though none of them worked for me from the box, given that the majority was for keras >1.2. 7. 7. Even keras 2.0 required some layer cropping to launch properly);
8. Pure tensorflow black-box implementation also exists (did not try it);
9. Relatively easy to understand why the architecture works, if you have basic understanding of how convolutions work.
- Disadvantages
1. Because of many layers takes significant amount of time to train;
2. Relatively high GPU memory footprint for larger images:
3. 640x959 image => you can fit 4-8 images in one batch with 6GB GPU;
4. 640x959 image => you can fit 8-16 images in one batch with 12GB GPU;
5. 1280*1918 => you can fit 1-2 images in one batch with 12GB GPU;
6. Is less covered in blogs / tutorials than more conventional architectures;
7. The majority of Keras implementations are for outdated Keras versions;
8. Is not standard to have pre-trained models widely available (it's too task specific).

## Some fun reference:
- SSD paper: https://arxiv.org/pdf/1512.02325v5.pdf
- SSD github: https://github.com/weiliu89/caffe/tree/ssd
- SSD training: https://www.kdnuggets.com/2017/11/understanding-deep-convolutional-neural-networks-tensorflow-keras.html/2
- Unet paper: https://arxiv.org/pdf/1505.04597.pdf
- Unet page: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
- Unet introduction: https://spark-in.me/post/unet-adventures-part-one-getting-acquainted-with-unet
