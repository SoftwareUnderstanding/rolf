## Multi-object detection by lightweight CNN on embedded system.

#### Project abstract:

1. I mainly use those lightweight CNNs to detect hand pose on raspberry pi, but they can detect other objects easily too.
2. Some lightweight CNNs In the table below are reference to github project and I improved them, and others are reference to some papers.
3. I have created some scripts to generate voc formate dataset for caffe, it help me to extend dataset fast.



| Lightweight CNNs | note                 | Supported System    |
| ---------------- | -------------------- | ------------------- |
| MobileNet-SSD    | Improved             | Ubuntu and Raspbian |
| MobileNetV2-SSD  | Haven't improved yet | Ubuntu              |
| SqueezeNet-SSD   | Significant improved | Ubuntu and Raspbian |
| ShuffleNet-SSD   | Improved             | Ubuntu with cuda    |
| Tiny-SSD         | Poor performance     | Ubuntu and Raspbian |



- **Platform:** 
  - Training: Ubuntu
  - Application: Raspbian



- **CNN Framwork:** Caffe



- **Dependencies:** Opencv



- **Reference CNN:**

  - SSD: 
    - Paper: https://arxiv.org/pdf/1512.02325.pdf
    - Github: https://github.com/weiliu89/caffe/tree/ssd
  - MobileNet:
    - Paper: https://arxiv.org/pdf/1704.04861.pdf
    - Github: https://github.com/tensorflow/models/tree/master/research/slim/nets
  - MobileNetV2:
    - Paper: https://128.84.21.199/pdf/1801.04381.pdf
    - Github: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
    - Blog: https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
  - SqueezeNet:
    - Paper: https://arxiv.org/pdf/1602.07360.pdf
    - Code: https://github.com/DeepScale/SqueezeNet
  - ShuffleNet:
    - Paper: https://arxiv.org/pdf/1707.01083.pdf
    - Code: https://github.com/farmingyard/ShuffleNet ; https://github.com/MG2033/ShuffleNet

  

- **Reference github:**

  - https://github.com/weiliu89/caffe
  - https://github.com/chuanqi305/MobileNet-SSD
  - https://github.com/chuanqi305/MobileNetv2-SSDLite
  - https://github.com/chuanqi305/SqueezeNet-SSD







