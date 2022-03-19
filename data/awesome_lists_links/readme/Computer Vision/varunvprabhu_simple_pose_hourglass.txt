# simple_pose_hourglass
This is an implementation of an already existing work on Pose Estimation by Walid Benbihi (https://github.com/wbenbihi/hourglasstensorlfow)
which in turn is based on the hourglass model by Alejandro Newell et al (http://arxiv.org/abs/1603.06937)

This work was a bit of an experiment and the thing I did differently is with the ground truth inputs. The original implementation used a
gaussian heatmap to mark the joints. I used the pose data to build an input where the torso was green, legs were red and the hands were
blue, all with a black background. The idea was to infer an image and get a prediction in the same style as the ground truth input.

There are 2 jupyter notebooks: one for training and the other for inference. The MPII dataset was used for training.

Code used: Tensorflow 1.9 for GPU along with the requisite CUDA and CuDNN libraries on a Win10 machine with Ryzen 1600, 
16GB RAM and NVIDIA 1070.

Training over 25000 images (at 256x256x3) took 1000s for one epoch (batch size of 16). Network was trained for 60 epochs. Learning rate
was set at 2.5e-4. 4 hourglass stacks were used.



Here are the outputs. The gifs were made from frames captured and inferred in real time from a webcam feed.

![1](https://github.com/varunvprabhu/simple_pose_hourglass/blob/master/output/OUTPUT.png)

![2](https://github.com/varunvprabhu/simple_pose_hourglass/blob/master/output/test_1.gif) ![3](https://github.com/varunvprabhu/simple_pose_hourglass/blob/master/output/test_4.gif) ![4](https://github.com/varunvprabhu/simple_pose_hourglass/blob/master/output/test_5.gif)

More details over here: http://varunvprabhu.blogspot.com/2018/09/simple-pose-estimation-with-hourglass.html
