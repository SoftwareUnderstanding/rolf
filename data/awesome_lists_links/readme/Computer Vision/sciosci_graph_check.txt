# Graphical integrity issues in open access publications: detection and patterns of proportional ink violations

Academic graphs are essential for communicating scientific ideas and results. Also, to turthfully reflct data and results, visualization reserchers have proposed several principles to guide creation process. In this work, we created a deep learning-based method, focusing on bar charts, to measure violations of the proportional ink principle and the specific rules are: a bar chart’s y-axis should start from zero, have one scale, and not be partially hidden (Bergstrom & West, 2020; Tufte, 2001). Based on the 5-folds cross validation, the AUC of the method is 0.917 with 0.02 standard deviation, which means the model is capable of distinguishing graphs with or without graphical integrity issues and low standard deviation relfects stable performance. The precision is 0.77 with 0.0209 standard error. [Paper Link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009650)

# Requirement
1. YoloV4 (Bochkovskiy et al., 2020)
```bash
git clone https://github.com/AlexeyAB/darknet.git
```
   - Follow the instruction in https://github.com/AlexeyAB/darknet.git to install YoloV4
2. Stroke Width Transform (Epshtein et al., 2010)
```bash
git clone https://github.com/sunsided/stroke-width-transform.git
```
   - Use convert_swt.py to do the transformation.
3. Reverse-Engineering Visualizations(REV) (Poco & Heer, 2017)
```bash
git clone https://github.com/uwdata/rev.git
```
   - Move REV.py inside the rev.git folder before doing REV. Follow the instruction in the notebook to implement.

# Method 
1. Compound Figure Classification: We used CNN (Resnet-101v2, pre-trained on ImageNet) with fine-tune to classify figures into compound and non-compound figures.
2. Subfigure Seperation: After classification, we would seperate compound figures by CNN (YOLO v4, pre-trained on MS COCO dataset) we trained to localize subfigures in compound figures. Then, did the seperation based on the result from CNN.
3. Image Classification: We focus on bar charts in this study, so we collected diagnostc figures from IMageCLEF2016 competition (García Seco de Herrera et al., 2016) and fine-tuned a CNN (Resnet-101v2, pre-trained on ImageNet) to classify figures into categroies. (Ex: bar charts, line charts, scatter chats, ......)
4. Text Localization: We fine-tuend a CNN (YOLO v4,  pre-trained on MS COCO dataset) to detect or localize texts on academic figures, prepocessed with Stroke Width Transformation (Bochkovskiy et al., 2020; Epshtein et al., 2010).
5. Text Recognition: Using fine-tuned Tesseract model of English texts to recognize the content of texts from text localization.
6. Text Role Classification: Using open-source model to predict the role of texts on academic figures, basecd on the geometric information of texts (Poco & Heer, 2017).
7. Feature Engineering: Before the feature engineering, we had a preprocessing process to correct some misclassifications. Then, extracing 7 features (see below chart) from figures to train the detector.

| Feature Description | Reason |
| ----------- | ----------- |
| The value of the lowest y-axis label on the y-axis (detected or inference from y-axis) | he lowest y-axis label should be zero |
| The increasing rate between each pair of y-axis labels | The scales of y-axis should be consistent across each pair of neighbor y-axis labels |
| If we need to inference the lowest text on the y-axis | If the lowest label on the y-axis is far from the x-axis, then we might ignore the actual lowest label on the y-axis |
| If the y-axis has a mix of integer and float number | Tesseract might not perform well with float number, and thus the increasing rate in the y-axis might not be accurate |
| The probability of being texts | We prefer texts with a higher probability of being texts |
| The OCR confidences  of texts on the y-axis | We prefer predictions of the content of texts with a higher confidence |
| The probability of being bar charts | Our classifier only classifies bar charts. Thus we prefer figures with a high probability of being bar charts |

## Method FlowChart
<img src="https://github.com/sciosci/graph_check/blob/main/images/flowchart.png" alt="drawing" width="600"/>

# Example
<p float="left">
  <img src="https://github.com/sciosci/graph_check/blob/main/images/Example1.png" width="400" />
  <img src="https://github.com/sciosci/graph_check/blob/main/images/Example2.png" width="400" /> 
</p>
<p float="left">
  <img src="https://github.com/sciosci/graph_check/blob/main/images/Example3.png" width="400" />
  <img src="https://github.com/sciosci/graph_check/blob/main/images/Example4.png" width="400" /> 
</p>
The y-axis of upper two graphs does not start from zero and there are truncations in lower two graphs. Therefore, these graphs would be annotated graphical integrity issues.

# License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a> \
(c) Han Zhuang, Tzu-Yang Huang, and Daniel Acuna 2020 - 2021 Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)  \
[Science of Science + Computational Discovery Lab](https://scienceofscience.org/) in the School of Information Studies at Syracuse Univeristy.

# Reference
- Bergstrom, C. T., & West, J. D. (2020). Calling Bullshit: The Art of Skepticism in a Data-Driven World (Illustrated Edition). Random House.
- Tufte, E. R. (2001). The visual display of quantitative information (Vol. 2). Graphics press Cheshire, CT.
- Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. ArXiv:2004.10934 [Cs, Eess]. http://arxiv.org/abs/2004.10934
- Poco, J., & Heer, J. (2017). Reverse-engineering visualizations: Recovering visual encodings from chart images. Computer Graphics Forum, 36, 353–363.
- Epshtein, B., Ofek, E., & Wexler, Y. (2010). Detecting text in natural scenes with stroke width transform. 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2963–2970. https://doi.org/10.1109/CVPR.2010.5540041
