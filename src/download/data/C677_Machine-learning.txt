# AI Application
- Hyojung Chang
- Jonggeun Park
- Minjoo Lee

This is an AI application that diagnoses lung diseases (cancer, nodules, covid-19 pneumonia).
Comparing the model using the PyTorch library and the model using the detectron2, we selected the first model with high accuracy.

Both models used 'Faster R-CNN' algorithm.

- Faster R-CNN?
```
...

Our object detection system, called Faster R-CNN, is composed of two modules. 
The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector [2] that uses the proposed regions. The entire system is a single, unified network for object detection (Figure 2). 
Using the recently popular terminology of neural networks with ‘attention’ [31] mechanisms, the RPN module tells the Fast R-CNN module where to look. 
In Section 3.1 we introduce the designs and properties of the network for region proposal. 
In Section 3.2 we develop algorithms for training both modules with features shared.

...omitted below

```

reference : https://arxiv.org/pdf/1506.01497.pdf
