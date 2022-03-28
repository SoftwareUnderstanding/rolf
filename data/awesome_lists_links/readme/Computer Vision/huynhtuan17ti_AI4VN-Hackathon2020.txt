# AI4VN-Hackathon2020
AI Hackathon Competition in VietNam, leaderboard of preliminary round: http://smartcitizen.ml/  

# Overview the competition
The problem given is image classcification  
There are 7 classes (not include class 0):  
+ Class 0: there is no event  
+ Class 1: fallen tree  
+ Class 2: fire  
+ Class 3: flooding  
+ Class 4: bad road  
+ Class 5: traffic jam  
+ Class 6: garbage  
+ Class 7: traffic accident  

# Our solution  
- We used EfficientNetB6 model as a final solution.  
- Our data has up to 32000 images for 8 classes (include class 0).  
- We didn't use multiclass for this problem. Instead, we used multilabel, in order to predict class 0 efficiently.  
- To predict class 0, we used a threshold to do that. If the max probability of all classes (class 1 to 7) has lower than the threshold, we label it zero.  
- To find the fitness threshold, we tuned it by taking average of all best thresholds in evaluating 5 folds.  
- We had used test time augmentation in preliminary round. It helped us gain around 0.02 in the leaderboard. But in the final, because it take so much time to predict so we didn't used it.  

# Result
#### We got in top 20 in the preliminary round.  
![](images/preliminary_result.PNG)  
#### In the final we got 6th place.  
![](images/final_result.PNG)  

# References
More details about EfficientNet: https://arxiv.org/abs/1905.11946  
More details about multilabel: https://en.wikipedia.org/wiki/Multi-label_classification  
More details about k-fold cross-validation: https://machinelearningmastery.com/k-fold-cross-validation/  
More details about test time augmentation: https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/  
