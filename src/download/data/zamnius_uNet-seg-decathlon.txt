# uNet-parper-reprodct

#### NOTE: THE JUPITER NOTEBOOKS CAN ALSO BE ACCESSED WITH GOOGLE COLAB LINKS PROVIDED BELOW ####

A custom designed U-Net achitecture shows generalisability when applied to 2 different semantic segmentation tasks. It can segment the both datasets, separateley, without human interaction.

The architecture is defined accourding to U-net architecture paradigm (Ronneberger et al., 2015)
https://arxiv.org/abs/1505.04597
The two datasets were taken from Medical Segmetation Decathlon Challenge
http://medicaldecathlon.com, Paper on the dataset: ttps://arxiv.org/abs/1902.09063

# Cardiac Dataset. The dice score of resulted U-net on unseen data is 80-84%
https://colab.research.google.com/drive/1ut4KfgxcQmM1bHDxrCoW8nZXxSythUdU?usp=sharing
Target: Left Atrium

Modality: Mono-modal MRI  

Size: 30 3D volumes (20 Training + 10 Testing)

Source: King’s College London

Challenge: Small training dataset with large variability

# Spleen Dataset The dice score of resulted U-net on unseen data is 79-91%.
https://colab.research.google.com/drive/1hKYOWHFvFIbIWsqnn7-DOKGu6LsTLTpY?usp=sharing
Target: Spleen

Modality: CT  

Size: 61 3D volumes (41 Training + 20 Testing)

Source: Memorial Sloan Kettering Cancer Center

Challenge: Large ranging foreground size
