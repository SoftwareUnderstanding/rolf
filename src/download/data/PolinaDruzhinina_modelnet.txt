# MobileNets
## Differences
- The first version bacome known for first using depth wise separable conv. It helps to significantly reduce the size and complexity of the model, besides fewer parameters and calculations have made Mobilenet especially useful for mobile and embedded applications. 
- MobileNetV2, based on the ideas of the first version, also introduces linear bottlenecks between layers and short connections into the architecture, which allow to accelerate training and increase accuracy 
- The latest, version V3, added squeeze and excitation layers to the original blocks presented in V2. According to the authors of the article, by using SE and h-swish in layers where the tensors are smaller, there is less delay and quality gain.

### Links to articles
- [ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications (https://arxiv.org/pdf/1704.04861.pdf)](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks  (https://arxiv.org/pdf/1801.04381.pdf)](https://arxiv.org/pdf/1801.04381.pdf)
- [ Searching for MobileNetV3 (https://arxiv.org/pdf/1905.02244.pdf)](https://arxiv.org/pdf/1905.02244.pdf)
Информация также изучалась на таких источниках, как:
-[https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa](https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa)
-[ https://medium.com/@lixinso/mobilenet-c08928f2dba7#:~:text=o%203.2%25%20more%20accurate%20on,MobleNet%20V2%20on%20COCO%20detection.&text=o%206.6%25%20more%20accurate%20compared,MobileNetV2%20model%20with%20comparable%20latency](https://medium.com/@lixinso/mobilenet-c08928f2dba7#:~:text=o%203.2%25%20more%20accurate%20on,MobleNet%20V2%20on%20COCO%20detection.&text=o%206.6%25%20more%20accurate%20compared,MobileNetV2%20model%20with%20comparable%20latency.)

 # Run
 
-requirements.txt - For installation:
```
pip install -r requirements.txt
```
- to run training use:

```
python train_run.py 
       -v (v1/v2/v3), default='v3'
       --mode (train/test) defolt = train
       --load (True/False) - for loading model
       --mobilenet (mobilenetv3.pth) -pretrained model
       -b --batch_size, default=64 - Batch size for training
       --num_workers, default=8, Number of workers used in dataloading
       --lr, --learning-rate default=0.01, - initial learning rate
       -epoch, --max_epoch default=100, - max epoch for training
       --save_folder, default='img/'
       --save_img, default=True, save test images
       --weight_decay, default=5e-4, Weight decay for SGD
       --momentum, default=0.999
```
For example:
```
python train_run.py -v v1 --mode test --load True --mobilenet mobilenetv1.pth
```
- to run validation and save image in folder:
```
python train_run.py -v v2 --mode test --load True --mobilenet mobilenetv2.pth
```


## possible improvements

- This implementation is a test one, in the future I would like to try not only simplified versions of architectures (used due to lack of capacity), try different parameters for lr, optimizer, regularizations (l1, l2), add a dropout at the end of the model, add more augmentations on the pictures. 
- It would also be nice to clean up the code and use the profiler more carefully, so that it does not give out an endless set of information that is difficult to process

