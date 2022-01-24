# WideResNet_Emotion
Implementing the wide residual network for detecting emotion.

## Installations
The project has been done using python 3.6. All the package required are mentioned in requirement.txt

## Motivation
The main motivation for doing this project is to understand how WideResNet works and to detect the accuracies on the [FER-2013 DataSet](https://www.kaggle.com/deadskull7/fer2013). The aim of the project is to tune the weights of the network to improve the current results.

## Using the File
[wideresnet.py](https://github.com/piyush9923/WideResNet_Emotion/blob/master/wideresnet.py) is the main file. It contains the entire architecture of the network. The file trains the model and save the learned weights.

[fnn.py](https://github.com/piyush9923/WideResNet_Emotion/blob/master/fnn.py) contains the code for training using Feed Forward Neural Newtork

## Conclusion
We were able to report an accuracy of 68% although there is scope of improvement.

## Licensing, Authors, Acknowledgements, etc:
For implementing the WideRes Net the follwing paper and code has been followed. 

Paper (v1): http://arxiv.org/abs/1605.07146v1 (the authors have since published a v2 of the paper, which introduces slightly different preprocessing and improves the accuracy a little).
Original code: https://github.com/szagoruyko/wide-residual-networks
