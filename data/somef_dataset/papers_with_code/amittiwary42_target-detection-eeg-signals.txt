# Target Detection in EEG Signals using Convolutional LSTMs

Code for the work carried out at CEERI, Pilani. The BCICIII_DatasetII was used for the EEG Signals. The resources used for the research are included as well.

##### As the data was skewed, we tried multiple approaches to remove the bias:

* raw_data: The dataset was undersampled to achieve parity
* oversampled: The dataset was oversampled
* retinanet: One of the labels was penalized using a focal loss during training as described in the Retinanet paper (https://arxiv.org/abs/1708.02002)

