#dnn-ssd-object-detection

•	In this I used the MobileNet SSD + deep neural network ( dnn ) module in OpenCV to build the object detector.

•	For more about SSD and MobileNets click on the given links::-

The object detection framework used is SSD: Single Shot MultiBox Detector - https://arxiv.org/abs/1512.02325

The base network which fits into the object detection framework is MobileNets: - https://arxiv.org/abs/1704.04861

•	The model used in this program is a Caffe version of the original TensorFlow implementation by Howard et al., We can therefore detect 20 objects in images (+1 for the background class), including airplanes, bicycles, birds, boats, bottles, buses, cars, cats, chairs, cows, dining tables, dogs, horses, motorbikes, people, potted plants, sheep, sofas, trains, and tv monitors.

•	In order to run this script, you’ll need to grab the required model and prototxt files.
Once you’ve extracted the files, open command prompt and navigate to downloaded code + model. From there, execute the following command::-
python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
