# pneumonia-rsna

Pneumonia data is from the Kaggle RSNA competition. I trained a densenet121 on the data for classification.

The following figures show the ROC plots for the validation and test splits. AUC of the validation split is 0.831 while that of the test split is 0.826.

![Figure](roc_valid_auc0p831.png)
![Figure](roc_test_auc0p826.png)

The following figures show the ROC plots for the validation and test splits with data augmentation during training. AUC of the validation data is 0.878 and that of the test data is 0.881. AUC improves signicantly with data augmentation.

![Figure](roc_valid_da_auc0p878.png)
![Figure](roc_test_da_auc0p881.png)

AUC of 0.881 is much better than the pneumonia AUC of 0.768 in the Stanford Chexnet paper (https://arxiv.org/pdf/1711.05225.pdf)

