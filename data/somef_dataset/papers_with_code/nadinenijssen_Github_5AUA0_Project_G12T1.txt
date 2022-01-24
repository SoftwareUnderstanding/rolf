# Project for the course 5AUA0
Group 12, Team 1: Kevin and Nadine

# One-shot multiple-object tracking using contrastive learning
## Summary
In multi-object tracking the leading paradigm is tracking-by-detection which is often a two step approach. Recent one-shot approaches have shown promising results that are able to run in real-time. One-shot models learn detections and appearance embeddings jointly. We built upon the one-shot method by changing the way the embeddings are trained. Softmax based features are trained by classifying the embedding feature map to the correspond track-IDs. We propose a pairwise loss that is able to learn embeddings without track-ID labels. On the MOT17 dataset we obtain competitive results with respect to the softmax based method.

## Paper
See [paper](5AUA0_Project_Group12_Team1.pdf).
