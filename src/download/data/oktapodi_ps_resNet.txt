# Facial Recognition using ResNets

## Network Architecture
A CNN built for face recognition taks, using the ResNet Architecture

## Data Preprocessing
While working on MobileNetV2 I read online about the preprocessing step done to the images. I decided to normalize using mean = [0.406, 0.456, 0.485] and std = [0.229, 0.224, 0.225]. Also randomly flipped images on the horizontal axis.

## Hyper Parameters
The model was initialized with the following hyper parameters 1. Epochs - 15
2. Learning Rate - 0.001
3. Weight Decay - 10^-5
4. Hidden Layers - [64, 128, 256, 512]
5. Loss Function - Cross Entropy Loss
6. Optimizer - Stochastic Gradient Decent 7. Momentum - 0.9
8. Learning Rate Scheduler [3]
1. Step Size - 7 2. Gamma - 0.1

## Key Takeaways
I tried cross entropy loss, Arc Face etc. [4, 5] but the loss functions did not have as great an effect as tuning the overall network architecture. With better data preprocessing and maybe larger hidden layers I could’ve got even better results.

## References
[1] “Mobile Net Version 2”, Matthijs Hollemans, 22 April, 2018, https://machinethink.net/blog/
mobilenet-v2/, Last Accessed 18 March 2019
[2] MobileNet V2 Quantized, Lu Fang, Oct 22, 2018, https://github.com/caffe2/models/tree/ master/mobilenet_v2_quantized, Last Accessed 18 March 2019
[3] “TORCH.OPTIM.LR_SCHEDULER”, PyTorch Documentation, https://pytorch.org/docs/ stable/_modules/torch/optim/lr_scheduler.html, Last Accessed 18 March 2019
[4] “ArcFace: Additive Angular Margin Loss for Deep Face Recognition”, Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou, Feb 9, 2019, https://arxiv.org/abs/1801.07698, Last Accessed 18 March 2019
[5] “A Performance Comparison of Loss Functions for Deep Face Recognition”, Yash Srivastava, Vaishnav Murali, Shiv Ram Dubey, Jan 1, 2019, https://arxiv.org/abs/1901.05903, Last Accessed 18 March 2019