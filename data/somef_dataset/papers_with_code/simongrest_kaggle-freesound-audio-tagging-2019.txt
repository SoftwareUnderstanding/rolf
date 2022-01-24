# Kaggle Freesound Audio Tagging Competition
## 44th place silver medal solution

Competition details can be found here:
https://www.kaggle.com/c/freesound-audio-tagging-2019

My kaggle profile: https://www.kaggle.com/simongrest

## Task
The task in this competition is drawn from the second task from the Detection and Classification of
Acoustic Scenes and Events (DCASE) 2019: http://dcase.community/challenge2019/task-audio-tagging

<img src="http://d33wubrfki0l68.cloudfront.net/98c159a16704dac8b3861c3a5c7672bb5ce15656/696eb/images/tasks/challenge2019/task2_freesound_audio_tagging.png" width="50%"/>

The task is a multi-label classification problem, audio samples need to be tagged with one or more of 80 labels drawn from Google's AudioSet Ontology.

## Datasets
This competition is based on two datasets:

### Curated dataset
A dataset with about 5000 audio files varying in length from 0.3 to 30 seconds, labelled by hand.

### Noisy dataset
A larger dataset with approximately 20,000 audio files drawn from videos on Flickr and labelled automatically based on tags and other meta-data.

The key part of this competition is to figure out how to use effectively use the larger noisy dataset. In order to do so, one needs to address the noisy labels and differences in the domain between the datasets. From the DCASE task description:

> The main research question addressed in this task is how to adequately exploit a small amount of reliable, manually-labeled data, and a larger quantity of noisy web audio data in a multi-label audio tagging task with a large vocabulary setting. In addition, since the data comes from different sources, the task encourages domain adaptation approaches to deal with a potential domain mismatch.

## Sound preprocessing
For this solution, audio files were transformed to a spectral or frequency representation. The mel-scale is used for the spectrograms. Instead of a decibel front-end a PCEN front-end is used - see https://arxiv.org/abs/1607.05666

<img src="images/PCEN and DB.png" width="70%"/>

## Network Architecture
This solution borrows heavily from an approach of the winners of the 2018 competition. Their technical report is here:
http://dcase.community/documents/challenge2018/technical_reports/DCASE2018_Jeong_102.pdf

Their implementation here:
https://github.com/finejuly/dcase2018_task2_cochlearai

Essentially the architecture is a densely connected CNN with Squeeze-Exitation blocks.

<img src="images/high_level_arch.png" width="25%"/><img src="images/block_arch.png" width="70%"/>

The Squeeze and Exitation block described here: https://arxiv.org/abs/1709.01507
<img src="images/se_arch.png" width="70%"/>

## Data augmentation
The following data augmentations were used:
1. Random selection of 4 second subset of audio clip
2. MixUp - https://arxiv.org/abs/1710.09412
3. Mild zooming and warping
4. Test Time Augmentation

## Training Approach
1. Train a CNN on the curated set using 6-fold cross validation
2. Use this model to predict the entire noisy dataset
3. Pick a class-balanced subset of examples from the noisy dataset where the predictions are reasonably good
4. Finetune the model trained on the curated set on the combination of curated and selected noisy subset

## Submission
This competition was run as a kernels only competition. The kernel had a maximum runtime of one hour. Available disk space was limited to ~5Gb and there was ~14Gb of memory available.

In order for the kernel to run more quickly spectrograms are loaded into memory in a shared dictionary. Memory usage is monitored and once it reaches 95% usage, spectrograms are written to disk instead of being added to memory.

## Result
44th of 808 participants - Top 5% - Silver Medal :)

https://www.kaggle.com/c/freesound-audio-tagging-2019/leaderboard


