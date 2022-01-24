# human_pose_estimation
# Introduction 
The recent methods of estimating the human posture in two-dimensional space based on deep learning have shown better applicability and results than before. However, the problem also faced many different challenges such as in crowds, resolution, lighting, ... In this project, I analyze and evaluate pros and cons of the article “Pifpaf: Composite Fields For Human Pose Estimation ”, the author has focused on solving the challenge of occluded and low resolution. I analyze on different datasets: COCO dataset, in addition I analyze errors on 1000 MPII images, 2000 images of sports dataset collected by us. Thereby having a more general view of the article, and thereby giving directions to develop research to improve the article. 

This project based on paper : https://arxiv.org/abs/1903.06593 and code: https://github.com/vita-epfl/openpifpaf

I analyze and evaluate on datasets:4000 images COCO test-dev,2000 images sports,1000 images MPII datasets:

Because the author focused on solving the challenge on low resolution and obscuration. I resized the images to 3 different resolutions: 256px, 321px, 641px. The author have suggested that 321px is the best but I want to experiment with 256px more, then I can conclude that 321px is the best or not.

Dataset: https://drive.google.com/drive/folders/19xFqlgraUi7BZp9VgBUY_UfC6u4gYNyY?usp=sharing

Analysis: https://drive.google.com/drive/folders/1BUkAbabOqjFUhi2_RlYds0vFS7Wg4ryB?usp=sharing

# PifPaf method

<img width="1200" alt="Screen Shot 2020-10-19 at 3 39 09 PM" src="https://user-images.githubusercontent.com/43095377/96421698-470a2500-1221-11eb-88d8-8a14c96892e2.png">

Source: on PifPaf paper

# Result
This folder contains images error about wrong connection, missing part,... : https://drive.google.com/drive/folders/1z0SbuqVmpruYGFVTXLKjQKRj45kSkVqj?usp=sharing 

Error(image) on 1000 images MPII (statistical)

<img width="1440" alt="Screen Shot 2020-10-19 at 3 28 50 PM" src="https://user-images.githubusercontent.com/43095377/96420687-ffcf6480-121f-11eb-95d4-ccf3d1698819.png">

Error(image) on 1000 images MPII (chart)

<img width="1439" alt="Screen Shot 2020-10-19 at 3 29 00 PM" src="https://user-images.githubusercontent.com/43095377/96420793-21c8e700-1220-11eb-9aa1-2d84f1028e8e.png">

Error(image) on 2000 images sport (statistical)

<img width="1438" alt="Screen Shot 2020-10-19 at 3 32 52 PM" src="https://user-images.githubusercontent.com/43095377/96421024-753b3500-1220-11eb-8f6c-617ea0e2f0b5.png">

Error(image) on 2000 images sport (chart)

<img width="1425" alt="Screen Shot 2020-10-19 at 3 33 01 PM" src="https://user-images.githubusercontent.com/43095377/96421111-913ed680-1220-11eb-9632-86f55f465d1f.png">

Error(image) on 4000 images COCO test-dev (statistical)

<img width="1437" alt="Screen Shot 2020-10-19 at 3 32 36 PM" src="https://user-images.githubusercontent.com/43095377/96421256-c21f0b80-1220-11eb-85c3-84c8b93f83b3.png">

Error(image) on 4000 images COCO test-dev (chart)

<img width="1427" alt="Screen Shot 2020-10-19 at 3 32 44 PM" src="https://user-images.githubusercontent.com/43095377/96421305-d19e5480-1220-11eb-97d1-4532d75af8ef.png">

# Summary

- The wrong connection and lack joint error is 256px smaller than 321px. Sometimes there is missing part.

- In terms of time, 256px is almost twice as fast as 321px.

- PifPaf is often wrong in the wrong connection, missing part error. In addition, MPII and COCO dataset 256px and 321px focus on other error.

- In sports environment, PifPaf focuses on wrong connection error (60-70% of total error).

