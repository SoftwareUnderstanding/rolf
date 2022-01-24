## Model Details

#### 14 observations (labels):
label_names = [ 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


3-Class model(0: negative, 1: positive, 2: uncertain):
https://arxiv.org/pdf/1901.07031.pdf

2-Class model(0: negative, 1: positive):
Choose the best from U-Zeros and U-Ones

U-Zeros model (0: negative, 1: positive, merge uncertain into negative for training):

U-Ones model (0: negative, 1: positive, merge uncertain into positive for training):
https://arxiv.org/abs/1705.02315  Wang, Xiaosong, Peng, Yifan, Lu, Le, Lu, Zhiyong, Bagheri, Mohammadhadi, and Summers, Ronald M. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. arXiv preprint arXiv:1705.02315, 2017.


#### Input:
224x224 image, convert to RGB, normalized based on the mean and standard deviation of training dataset of ImageNet


#### CNN Model:
densenet121 https://arxiv.org/abs/1608.06993
initialize parameters from the model pre-trained on ImageNet:
http://www.image-net.org/papers/imagenet_cvpr09.pdf 

Bottleneck Features:  1x1024 

——————————————————————————————
#### 3-Class Output:
dense layer: 14x3,  {p_0, p_1, p_2} on each label,  without Softmax(), since we use the loss function CrossEntropyLoss()

Loss Function (14-label, 3-class):
for 3 classes on each label, we use CrossEntropyLoss(), which includes Softmax(), Log() and NLLLoss(), where Log() and NLLLoss() return cross entropy. Then we take the average over 14 labels.

Final Output: apply Softmax() on only {p_0, p_1}, then use p_1 as the output of each label.
——————————————————————————————
#### 2-Class Output
dense layer: 14x1,  only {p_1} on each label,  without Sigmoid(), since we use the loss function BCEWithLogitsLoss()

Loss Function (14-label, 2-class):
we use BCEWithLogitsLoss(), which includes Sigmoid() and BCELoss(). Then we take the average over 14 labels.

Final Output:  apply Sigmoid() on {p_1}
——————————————————————————————


#### Optimizer
Adam: β1 = 0.9 and β2 = 0.999 as default
Learning rate: 1E-4
Decayed Factor: 10 / 2 epoch
Epoch Number: 6 or 4

#### Batch
Batch Size (based on the size of memory)
32 for 224x224, 16 for 320x320

#### Training Time
for 224x224: ~0.6 hour / epoch
for 320x320: ~1.3 hour / epoch
while Xception is some kind of slower than Densenet121, and Fractalnet is much slower.


#### ROC and PR in Valid dataset
use 2-class {p_0, p_1}, there is no uncertain,
we output ROC and PR for 14 observations



### AUC(ROC) Comparison 224x224, where U-Zeors/U-Ones label the uncertain as negative/positive:

| Type							| U0		| U1		| 2-Class	| 3-Class	|
| ----							| ----		| ----		| ----		| ----		|
| Atelectasis					| 0.75		| 0.81		| 0.82		| 0.75		|
| Cardiomegaly					| 0.84		| 0.79		| 0.82		| 0.85		|
| Consolidation					| 0.86		| 0.86		| 0.88		| 0.87		|
| Edema							| 0.93		| 0.93		| 0.94		| 0.93		|
| Pleural Effusion				| 0.92		| 0.92		| 0.93		| 0.91		|
| No Finding					| 0.91		| 0.90		| 0.91		| 0.91		|
| Enlarged Cardiomediastinum	| 0.62		| 0.50		| 0.59		| 0.59		|
| Lung Opacity					| 0.92		| 0.92		| 0.91		| 0.91		|
| Lung Lesion					| 0.32		| 0.64		| 0.83		| 0.18		|
| Pneumonia						| 0.73		| 0.70		| 0.80		| 0.70		| 
| Pneumothorax					| 0.91		| 0.89		| 0.91		| 0.92		|
| Pleural Other					| 0.96		| 0.87		| 0.92		| 0.93		|
| Fracture						| NaN		| NaN		| NaN		| NaN		|
| Support Devices				| 0.92		| 0.94		| 0.92		| 0.93		|


### Uncertain Method Selection for 2-Class, based on comparison between U-Zeros and U-Ones

| Type							| 2-Class	|
| --------						| ---------	|
| Atelectasis					| U-Ones	|
| Cardiomegaly					| U-Zeros	|
| Consolidation					| U-Zeros	|
| Edema							| U-Ones	|
| Pleural Effution				| U-Ones	|
| No Finding					| U-Zeros	|
| Enlarged Cardiomediastinum	| U-Zeros	|
| Lung Opacity					| U-Ones	|
| Lung Lesion					| U-Ones	|
| Pneumonia						| U-Zeros	|
| Pneumothorax					| U-Zeros	|
| Pleural Other					| U-Zeros	|
| Fracture						| U-Ones	|
| Support Devices				| U-Ones	|


### AUC(ROC) Comparison 320x320:

| Type							| CheXNet	| CheXNeXt				|   CheXpert			| Densenet(2)	| Densenet(3)	| Xception(2)	| Xception(3)	|
| ----							| ----		| ----					| ----					| ----			| ----			| ----			| ----			|
| Atelectasis					| 0.8094	| 0.862(0.825–0.895)	| 0.858(0.806,0.910)	| 0.81			| 0.73			| 0.83			| 0.78			|
| Cardiomegaly					| 0.9248	| 0.831(0.790–0.870)	| 0.854(0.800,0.909)	| 0.77			| 0.83			| 0.83			| 0.80			|
| Consolidation					| 0.7901	| 0.893(0.859-0.924)	| 0.939(0.908,0.971)	| 0.91			| 0.85			| 0.94			| 0.92			|
| Edema							| 0.8878	| 0.924(0.886-0.955)	| 0.941(0.903,0.980)	| 0.94			| 0.93			| 0.94			| 0.94			|
| Pleural Effusion				| 0.8638	| 0.901(0.868-0.930)	| 0.936(0.904,0.967)	| 0.93			| 0.92			| 0.93			| 0.94			|
| No Finding					| 			| 						| 						| 0.89			| 0.90			| 0.89			| 0.90			|
| Enlarged Cardiomediastinum	| 			| 						| 						| 0.54			| 0.57			| 0.51			| 0.49			|
| Lung Opacity					| 			| 						| 						| 0.93			| 0.92			| 0.92			| 0.92			|
| Lung Lesion					| 			| 						| 						| 0.77			| 0.47			| 0.21			| 0.60			|
| Pneumonia						| 0.7680	| 0.851(0.781-0.911)	| 						| 0.72			| 0.77			| 0.72			| 0.74			|
| Pneumothorax					| 0.8887	| 0.944(0.915-0.969)	| 						| 0.92			| 0.90			| 0.86			| 0.94			|
| Pleural Other					| 0.8062	| 0.798(0.744-0.849)	| 						| 0.96			| 0.95			| 0.96			| 0.98			|
| Fracture						| 			| 						| 						| NaN			| NaN			| NaN			| NaN			|
| Support Devices				| 			| 						| 						| 0.94			| 0.94			| 0.94			| 0.94			|

### AUC(PR) Comparison:

| Type							| CheXpert	| U0-224	| U1-224	| Des(2)224	| Des(3)224	| Des(2)320 | Des(3)320	| Xception(2)	| Xception(3)	|
| --------						| ---------	| -------	| ------	| ------	| ------	| ------	| ------	| ----			| ----			|
| Atelectasis					| 0.69		| 0.62		| 0.68		| 0.71		| 0.60		| 0.71		| 0.56		| 0.74			| 0.64			|
| Cardiomegaly					| 0.81		| 0.77		| 0.70		| 0.75		| 0.76		| 0.69		| 0.72		| 0.74			| 0.72			|
| Consolidation					| 0.44		| 0.52		| 0.44		| 0.53		| 0.51		| 0.63		| 0.51		| 0.69			| 0.68			|
| Edema							| 0.66		| 0.75		| 0.77		| 0.82		| 0.78		| 0.81		| 0.74		| 0.82			| 0.80			|
| Pleural Effution				| 0.91		| 0.86		| 0.86		| 0.87		| 0.85		| 0.87		| 0.84		| 0.87			| 0.87			|
| No Finding					| 			| 0.44		| 0.49		| 0.43		| 0.50		| 0.43		| 0.45		| 0.41			| 0.49			|
| Enlarged Cardiomediastinum	| 			| 0.65		| 0.55		| 0.62		| 0.60		| 0.55		| 0.59		| 0.55			| 0.53			|
| Lung Opacity					| 			| 0.94		| 0.94		| 0.94		| 0.93		| 0.95		| 0.94		| 0.95			| 0.94			|
| Lung Lesion					| 			| 0.00		| 0.01		| 0.01		| 0.00		| 0.01		| 0.00		| 0.00			| 0.01			|
| Pneumonia						| 			| 0.09		| 0.09		| 0.13		| 0.10		| 0.09		| 0.11		| 0.10			| 0.14			|
| Pneumothorax					| 			| 0.19		| 0.30		| 0.19		| 0.39		| 0.46		| 0.36		| 0.33			| 0.37			|
| Pleural Other					| 			| 0.06		| 0.02		| 0.03		| 0.04		| 0.06		| 0.05		| 0.06			| 0.10			|
| Fracture						| 			| NaN		| NaN		| NaN		| NaN		| NaN		| NaN		| NaN			| NaN			|
| Support Devices				| 			| 0.91		| 0.94		| 0.90		| 0.90		| 0.93		| 0.94		| 0.91			| 0.94			|


## Challenges we met

1. We tried to use random crop and random horizontal flip in the data preprocessing, however the performance was worse than do nothing.
Since random cropping might drop some important part of the images, and views from the front or from the back are actually different. While center cropping, padding zeros, and resizing to square got the similar performances, here we simply resized to square

2. To reproduce the paper, we supposed to use 320x320 resolution in the begining. However, the limitation of computing resource made us decrease the resolution to 224x224 first. We spend most of the time on raising the performance under 224x224. After that, we modified our input to 320x320. One of the reason why we must use 320x320 is because the input size of Xception model must be larger than 299x299 (if we hard code the input size to 224x224 in Xception, then the pretrained model performs bad since it is pre-trained on ImageNet by 299x299 )

3. A "mistake" we made was that we did prediction by each image, so the ROC performance was much worse than the paper. After we modified the prediction by each study (use the maximum if more than 1 images provided), the performance became similar with the paper

4. In the valid dataset, there is only 0 or 1 positive case in Lung Lesion, Pleural Other, and Fracture. The uncertain choices we made on these 3 types are based on the models trained on a re-split dataset. The re-split process should split the dateset by patients, not studies

5. We did not tune the Fractalnet too much, since it is very deep and costs a long time for training. But the results we got show that Xception and Densenet1212 perform much better than Fractalnet




## Implementation

### Step 1
in ./

conda env create environment.yml

conda activate chexpert


### Step 2
in ./data/

unzip CheXpert-v1.0-small.zip, then ./data/ should be like this:

./data/ train/

		valid/

		train.csv

		valid.csv

modify the data path in datasplit.py, train.py if you need

To check the performance of the trained model, we only need: valid/ and valid.csv


### Step 3

#### To train a new model:

1. empty output/

2. in ./code2class/, run train.py, model will be saved in ../output/

#### To check the performance of a trained model:

1. if ./output/ is empty, move a model file ?.pth into output/.  Currently, the MyCNN.pth in ./output/ is a 320x320 2 class Xception model.

3. in ./code2class/, run roc.py, graphs (ROC, PR) will be saved in ../output/

( make sure the "transforms" in roc.py is consistant with the trained model. Here I did not upload the 3 class Xception, so you may not run roc.py in code3class)


