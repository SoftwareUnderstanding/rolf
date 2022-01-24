# SAGAN

## SAGAN - Sharpness-aware Low dose CT denoising using Conditional generative adversarial network

### Our goal is to reproduce SAGAN. However, only the pix2pix learning concept is applied.

---

## ![result figure](https://github.com/MYUNGJE/SAGAN/blob/yu/figures/Result%20figure.png)

## 1. Prerequistites

#### 1.1 NVIDIA GPU

#### 1.2 Python 3.x

## 2. Getting Started

### 2.1 Install [Python3.x](https://www.anaconda.com/distribution/#linux) (recommend using Anaconda)

### 2.2 Install python dependencies

```
> $pip install -r requirements.txt
```

### 2.3 Clone this repo

```
> $git clone https://github.com/MYUNGJE/SAGAN.git
> $cd SAGAN
```

### 2.4 Prepare Dataset

#### 1) The piglet dataset we used in the SAGAN paper is now open for download in [here](http://homepage.usask.ca/~xiy525/project/low_dose_ct/)

#### 2) The file extension of this dataset is DICOM. Therefore, the extension should be converted to PNG. We used [mritopng](https://github.com/danishm/mritopng) to convert images.

#### 3) Dataset Diretory Hierarchy

#### - CT scans of a deceased piglet

##### --- Input(Dose + Recon) : 5% DOSE ASIR x820

##### --- Label(Dose + Recon) : 100% DOSE ASIR x820

```
│ data
│   ├── data
│   │   ├── train
│   │   │   ├── input x697 (randomly selected 123 images for Validation)
│   │   │   ├── label x697 (randomly selected 123 images for Validation)
│   │   ├── test
│   │   │   ├── input x123
│   │   │   ├── label x123
```

##### \*Validation dataset randomly selected 123 images from the train dataset and excluded them from the train dataset.

## 3. Model Architecture

### 3.1 Generator

![Generator](https://github.com/MYUNGJE/SAGAN/blob/yu/figures/Overview_generator.png)

### 3.2 Discriminator(Image)

![Discriminator](https://github.com/MYUNGJE/SAGAN/blob/yu/figures/Overview_discriminator_image.png)

## 4. Overview of Training Concept

![training concept](https://github.com/MYUNGJE/SAGAN/blob/yu/figures/Overview_training_concept.png)

## 5. How to use

### 5.1 Train

```
> $python sa_main.py --is_test=False --output_dir='./output_dir'
```

##### --output_dir (models are saved here.)

##### --is_test (if is_test==False, train.)

### 5.2 Test

```
> $python sa_main.py --is_test=True --test_dir='./test_inputs_dir' --model_dir='./model_dir' --output_dir='./test_result_output_dir' --mn=0 --fn=4000
```

##### --is_test (if is_test==True, test.)

##### --test_dir (test input files directory.)

##### --model_dir (saved models directory.)

##### --output_dir (test results directory, test results are saved here.)

##### --mn (model checkpoint -> all_model_checkpoint_paths index.)

##### --fn (folder name integer.)

### 5.3 Convert

```
> $python main.py --is_convert=True --test_dir='./test_inputs_dir' --model_dir='./model_dir' --output_dir='./test_result_output_dir' --mn=0
```

##### --is_convert (if is_convert==True, convert.)

##### --test_dir (test input files directory.)

##### --model_dir (saved models directory.)

##### --output_dir (conversion results directory, conversion results are saved here.)

##### --mn (model checkpoint -> all_model_checkpoint_paths index.)

## 6. Overview of source codes

```
│   src
│   ├── sa_Tensorflow_utils.py
│   ├── sa_convert.py
│   ├── sa_dataset.py
│   ├── sa_main.py
│   ├── sa_model.py
│   ├── sa_solver.py
│   ├── sa_utils.py
```

##### **-- sa_Tensorflow_utils.py** contains the layer elements that make up the network.

##### **-- sa_convert.py** outputs a generated image by inputting a single image or a folder containing many images (evaluation result is omitted)

##### **-- sa_dataset.py** configures the datasets used for train, validation, and test.

##### **-- sa_main.py** contains the flags of the source codes and executes the sa_solver.py code.

##### **-- sa_model.py** constructs and designs the model.

##### **-- sa_solver.py** designs the overall process.

##### **-- sa_utils.py** consists of utils needed in source codes.

## 7. Reference

### 7.1 SAGAN Paper

#### [https://doi.org/10.1007/s10278-018-0056-0](https://doi.org/10.1007/s10278-018-0056-0) , journal of Digital imaging, October 2018

#### [https://arxiv.org/pdf/1708.06453.pdf](https://arxiv.org/pdf/1708.06453.pdf), arxiv, October 2017

### 7.2 Github repository

#### [https://github.com/xinario/SAGAN](https://github.com/xinario/SAGAN), This repo provides the trained denoising model and testing code.

#### [https://github.com/xinario/defocus_segmentation], This repo provides the S network algorithm hints (See [lbpSharpness.py](https://github.com/xinario/defocus_segmentation/blob/master/lbpSharpness.py))

### 7.3 Acknowlegements

#### [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf), arXiv, November 2018

## 8. Experience Results

#### 1) We used the *PSNR and *SSIM for quantitative evaluation.

#### 2) The ratio of GAN loss to Segmentation loss was adjusted and tested.

##### \*PSNR (peak signal to noise ratio)

##### \*SSIM (The structural similarity)

#### 3) If you want to change the weights of the two loss values, you can change the values ​​of lambda1 (seg loss) and lambda2 (gan loss) in the flags.

#### 4) The three images used in the test are in test_imgs (gan _ 1 + seg _ 100) ... / input file and are the average of the three results.

#### 5) Table.1

|                    | PSNR                                  | SSIM                                  |
| ------------------ | ------------------------------------- | ------------------------------------- |
| **gan*1+seg*90**   | 38.67, 40.02, 39.89, 40.11, **40.35** | 0.964, 0.974, 0.972, 0.974, **0.974** |
| **gan*1+seg*100**  | 38.70, 38.80, 39.59, 40.48, **41.64** | 0.967, 0.969, 0.971, 0.975, **0.977** |
| **gan*2+seg*100**  | 38.34, 39.64, 38.98, 39.91, **41.20** | 0.967, 0.971, 0.97, 0.971, **0.975**  |
| **gan*5+seg*90**   | 37.74, 37.36, 38.67, 39.07, **39.05** | 0.948, 0.95, 0.961, 0.962, **0.96**   |
| **gan*5+seg*100**  | 37.81, 38.36, 39.51, 39.35, **40.59** | 0.955, 0.952, 0.967, 0.96, **0.97**   |
| **gan*10+seg*90**  | 37.09, 38.41, 38.55, 38.89, **39.41** | 0.947, 0.946, 0.946, 0.954, **0.965** |
| **gan*10+seg*100** | 37.13, 38.63, 38.83, 39.06, **39.76** | 0.938, 0.956, 0.956, 0.959, **0.958** |

##### \*The five values ​​in a cell are test results from 4000, 8000, 12000, 16000 and 20000 learning iterations from the left.

# cGANs-tensorflow-Python
