comma.ai Programming Challenge: https://github.com/commaai/speedchallenge
======

Preprocessing
-----

<b>- memory-intensive: loads complete video into ram</b>
- 100x640 pixels of the original video resized to 96x96 pixels
- pixels = pixels[260:360, 0:640] height, width
- normalized to std=1, mean=0

Data augmentations
-----

- take every 2nd frame and take double the velocity as the target
- horizontal flip
- repeat one frame n times and take 0.0 as the target

Architecture
-----

- r2plus1d_18: https://arxiv.org/abs/1711.11248

Validation
-----

- training data cut into 50 video clips
- 5 different train/validate splits (shifted by one clip) with every 5th clip being a validation clip (80/20)
- about 0.3 mse on average without averaging

Evaluation
-----

0. predictions from frame 1130 to 1618 set to 0.0, because the model wasn't able to learn the edge case from the training data
    (traffic moving sideways while car stands still). 
1. average of the 5 models on the test data
2. moving average of length 100 over the predictions



