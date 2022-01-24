# Classification network
Resnet18 with the last fully connected layer replaced with custom
fully connected layer. We go from (512, 1000) to (512, 1) for binary classification.

Train accuracy = 100%
Test accuracy = 100%

| Data split          |  Accuracy (%) |
| ------------------- |:-------------:|
| Train               |  100          |
| Validation          |  100          |


# Regression Network
## Train, validation and test set
I save the model ouputs and ground truth (target) in 3 json files:

* regression_model_train_outputs.json
* regression_model_validation_outputs.json
* regression_model_test_outputs.json

with the following format:
```json
{
    "image_path": {
        "target"      : "[x1, y1, x2, y2, ..., x22, y22]",
        "predictions" : "[x1, y1, x2, y2, ..., x22, y22]"
    }
}
```
There should be 1451, 483, and 486 Train, validation and test samples in each json file.

## Bad data
Since we don't have *targets* for this, I only output the predicted coordinates. Since images may have needed to be flipped, I employ the classification net to make that decision. The saved file is as follows.

* regression_model_bad_outputs.json

with format:

```json
{
    "image_path": {
        "predicated direction"      : "0 if left else 1 if right",
        "predictions"               : "[x1, y1, x2, y2, ..., x22, y22]"
    }
}
```

# Unet Idea
I trained a nested Unet (https://arxiv.org/pdf/1807.10165.pdf) on images of size 128x128, but all landmarks are saved as normalised image coordinates. As for predicting the landmarks using image processing techniqes: I tried overcompensating but still miss landmarks in a few images (11,2, and 4 for train, test and validation).
## Train, validation and test set
I save the model ouputs and ground truth (target) in 3 json files:

* train_unet_landmarks.json
* validation_unet_landmarks.json
* test_unet_landmarks.json

With format:

```json
{
    "image_path": {
        "target"            : "[x1, y1, x2, y2, ..., x22, y22]",
        "predictions"       : "[x1, y1, x2, y2, ... ]",
        "Number_landmarks"    : "n",
        "mask_path"           : "path for segmentation"
    }
}
```
Note that predictions is of length 2*Number_landmarks.

## Bad data
The process is similar to the one above. There is an additional field that contains the predicted direction so that an image or predictions can be flipped when displaying result. We have the file

* bad_unet_landmarks.json

```json
{
    "image_path": {
        "predicted_direction"        :  "0 if left else 1 if right",
        "predictions"                : "[x1, y1, x2, y2, ... ]",
        "Number_landmarks"           : "n",
        "mask_path"                  : "path for segmentation"
    }
}
```
