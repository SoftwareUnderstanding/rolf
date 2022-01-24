# CS492(I) CV Project

KAIST CS492(I) Special Topics in Computer Science[Deep Learning for Real-World Problems]

Authorized [DonghwanKim](https://github.com/DonghwanKIM0101)

Authorized [SeungilLee](https://github.com/ChoiIseungil)

-----------

The model output of FixMixMatch and ThresholdMixMatch

[Google Drive link](https://drive.google.com/drive/folders/1ScaF1vOAzG5SH8tRDENPIhInFcScjaBA?usp=sharing)

FixMixMatch_np.pt and FixMixMatch_p.pt are each for FixMixMatch model in non-pretrained and pretrained option
ThresholdMixMatch_np.pt and ThresholdMixMatch_p.pt are each for ThresholdMixMatch model in non-pretrained and pretrained option

# Table of contents
1. [Summary](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#summary)
2. [Method](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#method)

    2.1 [Threshold](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#threshold)

    2.2 [Data Augmentation](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#data-augmentation)

3. [Result](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#result)
4. [Conclusion](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#conclusion)
5. [Reference](https://github.com/DonghwanKIM0101/CS492I_CV/blob/main/README.md#reference)

# Summary

It is project in KAIST CS492(I) course. With NSML of NAVER, implement shopping item object detection model. 

![Alt text](Image/0a5e810ae2cbbf0bdbce393ed8209498.jpg)
![Alt text](Image/0a70b8806168e481d63f8331bbdf00f8.jpg)

These are the example of data.
Our team's approach is to exploit [FixMatch](https://arxiv.org/pdf/2001.07685.pdf) to [MixMatch](https://arxiv.org/pdf/1905.02249.pdf).

# Method

## Threshold

Threshold is one of main concept of FixMatch.

<img src="Image/threshold.png" width="450px"></img><br/>
> https://arxiv.org/pdf/2001.07685.pdf

By using threshold while guessing pseudo label, the model only learn for confident unlabeled data.
Original method use fixed threshold value, 0.95. Compared to original method, our team have to use non-pretrained model for this project.
We suggest new concept threshold scheduling.

<img src="Image/threshold_scheduling.png" width="450px"></img><br/>

In the graph, X-axis is current_epoch/total_epoch and Y-axis is the probability that unused unlabeled data.
For first epoch, the model learn the most confident 30% unlabeled data, and for last epoch, the model learn all of the unlabeled data. 

## Data Augmentation

FixMatch uses both weakly augmented data and strongly augmented data.

<img src="Image/data_augmentation.png" width="450px"></img><br/>
> https://arxiv.org/pdf/2001.07685.pdf

For weak data augmentation, Crop, Horizontal Flip, and Vertical Flip
For strong data augmentaion, Crop, Horizontal Flip, Vertical Flip, Rotation, Color Jitter, and Cutout

# Result

To check our own model, compare MixMatch, ThresholdMixMatch, FixMixMatch.
ThresholdMixMatch is MixMatch with threshold scheduling,
FixMixMatch is MixMatch with threshold scheduling, weak and strong data augmentation.
We use DenseNet121 for all tests.

<img src="Image/avg_top1_np.png" width="300px"></img><br/>
<img src="Image/avg_top5_np.png" width="300px"></img><br/>

For non-pretrained model, ThresholdMixMatch shows the best result and FixMixMatch shows the worst result.

<img src="Image/avg_top1_p.png" width="300px"></img><br/>
<img src="Image/avg_top5_p.png" width="300px"></img><br/>

For pretrained model, three models show similar result although FixMixMatch shows the worst result in average top1.

# Conclusion

We wanted to exploit FixMatch to MixMatch; FixMixMatch.
From the result, FixMixMatch does not show good result for non-pretrained model.
We guessed that it is because the strong data augmentation does not work well in non-pretrained model.

<img src="Image/conclusion.png" width="300px"></img><br/>

The graph proves our guess.

However, threshold scheduling improves the result.
Compared to original threshold concept, our new concept focuses more on non-pretrained model.
Also, by testing the models in pretrained option, we can get FixMatch works well in pretrained option but does not in non-pretrained option.

# Reference
D Berthelot, N Carlini, I Goodfellow, N Papernot, A Oliver, CA Raffel, MixMatch: A Holistic Approach to Semi-Supervised Learning, 2019 
Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence, 2020





