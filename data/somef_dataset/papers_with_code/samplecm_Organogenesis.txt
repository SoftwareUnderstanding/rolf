# Organogenesis   (Version 1.0)

**GitHub Project:** [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis)

Organogenesis is an open source program for the auto-segmentation of medical CT image structures. The program features two types of neural network architectures: PyTorch implementations of a modified U-Net ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597)) and MultiResUNet ([10.1016/j.neunet.2019.08.025](https://doi.org/10.1016/j.neunet.2019.08.025)<span style="text-decoration:underline;"> ).</span> Pre-trained models currently available for download include the body, spinal cord, submandibular glands, and parotid glands. The program can be used for training models that contour all types of organs, and also using models for predicting contours and creating RTSTRUCT DICOM files. 

Organogenesis should **NOT** be used for clinical purposes. It is suitable only for research use. 

For training, a patient’s CT set consisting of DICOM files for each individual image slice is required, as well as the patient’s RTSTRUCT DICOM file containing the training contours. For predicting, all that is required is the patient’s CT set. 


## Model Statistics



<table>
  <tr>
   <td>Region of Interest
   </td>
   <td>Model Type
   </td>
   <td>Best Threshold
   </td>
   <td>F Score
   </td>
  </tr>
  <tr>
   <td>Body
   </td>
   <td>UNet 
   </td>
   <td>0.72
   </td>
   <td>0.99
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet 
   </td>
   <td>0.5
   </td>
   <td>0.99
   </td>
  </tr>
  <tr>
   <td>Spinal Cord
   </td>
   <td>UNet
   </td>
   <td>0.05
   </td>
   <td>0.83
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet
   </td>
   <td>0.8
   </td>
   <td>0.75
   </td>
  </tr>
  <tr>
   <td>Left Parotid
   </td>
   <td>UNet 
   </td>
   <td>0.52
   </td>
   <td>0.80
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet 
   </td>
   <td>0.43
   </td>
   <td>0.77
   </td>
  </tr>
  <tr>
   <td>Right Parotid
   </td>
   <td>UNet 
   </td>
   <td>0.2
   </td>
   <td>0.76
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet 
   </td>
   <td>0.14
   </td>
   <td>0.67
   </td>
  </tr>
  <tr>
   <td>Left Submandibular
   </td>
   <td>UNet
   </td>
   <td>0.22
   </td>
   <td>0.80
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet
   </td>
   <td>0.05
   </td>
   <td>0.72
   </td>
  </tr>
  <tr>
   <td>Right Submandibular
   </td>
   <td>UNet 
   </td>
   <td>0.39
   </td>
   <td>0.58
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>MultiResUNet 
   </td>
   <td>0.56
   </td>
   <td>0.70
   </td>
  </tr>
  <tr>
    <td>
      Right Tubarial
    </td>
    <td>
      UNet
    </td>
    <td>
      0.3
    </td>
    <td>
      0.65
    </td>
         
</table>


## Table of Contents

1. Introduction
2. Getting Started (General Information)
3. Program Structure
4. Folder Structure
5. Getting Started in Linux/MacOS
6. Getting Started in Windows
7. Examples


## Getting Started (General Information)

The program is written using version 3.8 of the python programming language. 

Download python version 3.8.10 from [https://www.python.org/downloads/](https://www.python.org/downloads/)

 Organogenesis requires the following list of dependencies to run. 



* Albumentations
    * [https://pypi.org/project/albumentations/](https://pypi.org/project/albumentations/)
* Matplotlib
    * https://pypi.org/project/matplotlib/
* NumPy
    * [https://pypi.org/project/numpy/](https://pypi.org/project/numpy/)
* OpenCV
    * https://pypi.org/project/opencv-python/
* Open3D
    * [https://pypi.org/project/open3d/](https://pypi.org/project/open3d/)
* Plotly
    * https://pypi.org/project/plotly/
* PyTorch
    * [https://pypi.org/project/torch/](https://pypi.org/project/torch/)
* SciPy
    * https://pypi.org/project/scipy/
* Shapely
    * [https://pypi.org/project/Shapely/](https://pypi.org/project/Shapely/)
* SSIM-PIL
    * [https://pypi.org/project/SSIM-PIL/](https://pypi.org/project/SSIM-PIL/)	

See the “Getting Started with Linux/MacOS” or "Getting Started in Windows" sections for instructions on quickly installing all dependencies. Note that PyTorch and Shapely have system-dependent installation instructions.

PyTorch uses Nvidia’s CUDA to perform computations on a device’s GPU. It is advisable to use a GPU for model training as it greatly increases computation speed. Neural networks require many matrix multiplications which can be performed much faster by parallelizing on a GPU. 

Follow the instructions at [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) to download the CUDA toolkit for Linux/Windows.

For detailed descriptions of each function and class within the program, please visit our [Functions and Classes Readme](https://github.com/samplecm/Organogenesis/blob/7908b34b143c95c05d00908ea1e9ca36eb32862e/Functions%20and%20Classes%20README.md).

## Program Structure

Organogenesis features a simple command line interface. To launch the command line in Windows open the Command Prompt and use the cd command to change the directory to where the Organogenesis folder is stored. Then, enter commands as shown below. In Linux/MacOS, enter commands into the terminal as shown below. If the program is launched without any optional arguments

```
$ python Menu.py
```

then the program will prompt the user for additional information to perform a desired task.

The optional arguments available for use with the program can be viewed within the terminal using the command

```
$ python Menu.py -h
```

which displays:

optional arguments:

 **-h, --help** <br /> show this help message and exit

**-o ORGANS [ORGANS ...], --organs ORGANS [ORGANS ...]** <br /> Specify organ(s) to train/evaluate a model for or predict/generate contours with. Include a single space between organs. Please choose from: body, brain, brainstem, brachial-plexus, chiasm, esophagus, globes, larynx, lens, lips, mandible, optic-nerves, oral-cavity, right-parotid, left-parotid, spinal-cord, right-submandibular, left-submandibular, all

**-f FUNCTION, --function FUNCTION** <br /> Specify the function to be performed. Options include: <br /> "Train": to train a model for predicting the specified organ, <br />  "GetContours": to obtain predicted contours for a patient, <br /> "BestThreshold": to find the best threshold for maximizing a model's F score, <br /> "GetEvalData": to calculate the F score, recall, precision, accuracy and 95th percentile Haussdorff distance for the given organ's model, <br /> "PlotMasks": to plot 2d CTs with both manually drawn and predicted masks for visual comparison

**--lr LR** <br /> Specify the learning rate desired for model training

**--epochs EPOCHS** <br /> Specify the number of epochs to train the model for

**--processData** <br /> True/False. True if patient DICOM data needs to be processed into training/validation/test folders

**--loadModel** <br /> True/False. True if a pre-existing model is to be loaded to continue training

**--dataPath DATAPATH** <br /> If data is not prepared in Patient_Files folder, specify the path to the directory containing all patient folders

**--preSorted** <br /> True/False. True if contours have been sorted into "good" and "bad" contour lists and the data should be processed into test/validation/training folders using them, False if not

**--modelType MODELTYPE** <br /> Specify the model type. UNet or MultiResUNet. If predicting with multiple organs, please enter the model types in the same order as the organs separated by a single space

**--dataAugmentation** <br /> True/False. True to turn on data augmentation for training, False to use non-augmented CT images

**--predictionPatientName PREDICTIONPATIENTNAME** <br /> Specify the name of the patient folder in the Patient_Files folder that you wish to predict contours for. Alternatively, supply the full path to a patient's folder

**--thres THRES [THRES ...]** <br /> Specify the pixel mask threshold to use with the model (between 0 and 1). If predicting with multiple organs, please enter the thresholds in the same order as the organs separated by a single space

 **--contoursWithReal** <br /> True/False. True to plot the predicted contours alongside manually contoured ones from the patient's dicom file, False to just plot the predicted contours

**--loadContours** <br /> True/False. True to attempt to load previously predicted or processed contours to save time, False to predict or process data without trying to load files

**--sortData** <br /> True/False. True if the patient list is to be visually inspected for quality assurance of the contours, False if confident that all contours are well contoured

**--dontSaveContours** <br /> True/False. Specify whether or not you would like the predicted contours saved to a DICOM file. True to not save predicted contours, False to save them

## Folder Structure

Organogenesis operates with a specific folder structure which organizes models, patient files, and other data. The folder structure can be downloaded on GITHUB, or can be set up in a UNIX environment by running the bash script “FolderSetup.sh” in the program’s directory. FolderSetup.sh can be found in the main repository of Organogenesis. See "Getting Started in Linux/MacOS" and "Getting Started in Windows" sections for details. 

If following the conventional folder structure, patient files should be placed in the Patient_Files folder. Each patient should have a folder within Patient_Files containing DICOM files for each of their CT images. An RTSTRUCT DICOM file with manually contoured structures should also be added for training. If organs are being predicted without an RTSTRUCT file present, one will be automatically generated. If the conventional folder structure is not being used, a directory containing this list of patient folders can be included as an argument --dataPath. In this case, the necessary folders will be automatically created inside the given data path if working in a Linux/MacOS environment. In Windows, folders will have to be added manually. Note that we recommend the conventional folder system for ease of use. 

All models trained will be saved inside the Models folder. Model hyperparameters, F scores, and Best Thresholds will also be saved to this folder. For details on downloading the Organogenesis pre-trained models refer to the "Optionally Download Pre-trained Models" section of "Getting Started in Linux/MacOS" or "Getting Started in Windows".

Before training/model predictions can be performed, DICOM data must be processed in order to create binary masks for training contours and CT images saved as numpy arrays. This data is saved in the Processed_Data folder. Inside this folder, there are numerous subfolders for different organs. Each organ has a folder called “organ” (with the name of the actual organ) which contains processed training CT images/binary masks saved together as an encoded 4d numpy array. Similarly, the “organ_Val” and “organ_Test” folders contain processed data to be used for validation and testing. The data is automatically split between the three when data is processed. The Area Stats subfolder contains statistical averages for the number of points/areas of contours of different organs. These statistics are used to assess the quality of a prediction and fix them through interpolation if necessary. 

The Loss History folder contains a subdirectory for each organ which contains the average validation set losses calculated after each epoch during model training. 

## Getting Started in Linux/MacOS

The following instructions detail how to download all files and dependencies for using Organogenesis. 

1. Download python version 3.8.10 from [https://www.python.org/downloads/](https://www.python.org/downloads/) 
2. If pip is not installed on your system, open the terminal and enter
    ```
    $ sudo apt install python3-pip
    ```
    
3. Download Organogenesis repository
    1. Go to [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis) 
    2. Click "Organogenesis (Version 1.0)" in Releases
    3. Double click to download "Source code (zip)"
    4. Locate the zip folder in your files
    5. Move the zip folder to desired directory 
    7. Extract the files from the zipped folder 
    8. Rename the folder to "Organogenesis"
   
4. Install General Dependencies
    1. Open the terminal and navigate to the program directory which contains “FolderSetup.sh”
    2. Give permission to execute FolderSetup.sh by using the command
        ```
        $ chmod 777 FolderSetup.sh
        ```
    6. Execute the script using the command
        ```
        $ ./FolderSetup.sh
        ```
        
5. Install Computer Specific Dependencies
    1. Install torch, torchvision, and torchaudio 
        1. Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 
        2. Choose the options as displayed in the image below for Pytorch Build, Your OS, Package, and Language
        ![Image](https://github.com/samplecm/Organogenesis/blob/7908b34b143c95c05d00908ea1e9ca36eb32862e/ReadMeImages/LinuxMacOSPyTorchInstallation.jpg)
        3. Choose a Computer Platform option based on the specifications of your GPU or CPU
        4. Copy the text from Run this Command 
        5. Paste text into the terminal and hit Enter
       
6. Optionally Download Organogenesis Pre-trained Models
    1. Go to [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis) 
    2. Click "Organogenesis (Version 1.0)" in Releases
    3. Double click to download “Models” folder 
    4. Move the “Models” folder to the “Organogenesis” folder
    5. Right click on Model folder and click “Extract Here”

        **Note:** This will replace the contents of the current “Models” folder. If you have already trained your own model, please remove it so that it is not overwritten. 

## Getting Started in Windows

The following instructions detail how to download all files and dependencies for using Organogenesis. 



1. Download python version 3.8.10 from [https://www.python.org/downloads/](https://www.python.org/downloads/) 
2. Download Organogenesis repository
    1. Go to [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis) 
    2. Click "Organogenesis (Version 1.0)" in Releases
    3. Double click to download "Source code (zip)"
    4. Locate the zip folder in File Explorer
    5. Move the zip folder to desired directory 
    7. Right click on the zip folder and click “Extract Here”
    8. Rename the folder to "Organogenesis"
        
3. Download Organogenesis Folder Structure
    1. Go to [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis) 
    2. Click "Organogenesis (Version 1.0)" in Releases
    3. Double click to download “Organogenesis_Folder_Setup.zip” 
    4. Locate the zip folder in File Explorer
    5. Right click on “Organogenesis_Folder_Setup.zip” and click “Extract Here”
    6. Move all folders within “Organogenesis_Folder_Setup” into the "Organogensis" folder
 
    **Note:** The contents of the Organogenesis folder should look like the image below.
    
    ![Image](https://github.com/samplecm/Organogenesis/blob/2f019648829b9c892d8dea6c62bed5ac5c2725fb/ReadMeImages/OrganogenesisFolderContents.jpg)

4. Install General Dependencies
    1. Open the Organogenesis folder in File Explorer
    2. Right click on any file within the folder and select “Properties”
    3. Copy the directory path from “Location”
    4. Open the Windows Command Prompt application
    5. Use the cd command to change the default directory to the directory path containing the Organogenesis folder
    ![Image](https://github.com/samplecm/Organogenesis/blob/7908b34b143c95c05d00908ea1e9ca36eb32862e/ReadMeImages/WindowsCommandPromptExample.jpg)
    6. Enter the command below into the command line
        ```
        $ pip install -r requirements.txt
        ```
    7. Leave the Command Prompt open for step 4
   
5. Install Computer Specific Dependencies
    1. Install torch, torchvision, and torchaudio 
        1. Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 
        2. Choose the options as displayed in the image below for Pytorch Build, Your OS, Package, and Language
        ![Image](https://github.com/samplecm/Organogenesis/blob/7908b34b143c95c05d00908ea1e9ca36eb32862e/ReadMeImages/WindowsPyTorchInstallation.jpg)
        3. Choose a Computer Platform option based on the specifications of your GPU or CPU
        4. Copy the text from Run this Command 
        5. Paste text into the terminal and hit Enter
    2. Install Shapely 
        1. Enter the command below into the command line
            ```
            $ pip install wheel 
            ```

            **Note:** the command prompt should display “Successfully installed wheel”

        2. Find out whether you are using Windows 32-bit or 64-bit by going to Settings => System => About => System Type
        3. Go to [https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely) to download the wheel corresponding to your computer’s specifications.
        4. If your computer uses Windows 64-bit, choose Shapely‑1.7.1‑cp38‑cp38‑win_amd**64**.whl 

            If your computer uses Windows 32-bit choose Shapely‑1.7.1‑cp38‑cp38‑win**32**.whl

        5. Open the folder where the Shapely file was downloaded to in File Explorer
        6. Right click on the Shapely file and select “Properties”
        7. Copy the directory path from “Location”
        8. Paste the directory path in the Command Prompt followed by “\” and the name of the file you downloaded For example: C:\Users\UserName\Downloads\Shapely‑1.7.1‑cp38‑cp38‑win_amd**64**.whl 

            **Note:** the command prompt should display “Successfully installed Shapely”

6. Optionally Download Organogenesis Pre-trained Models
    1. Go to [https://github.com/samplecm/Organogenesis](https://github.com/samplecm/Organogenesis) 
    2. Click "Organogenesis (Version 1.0)" in Releases
    3. Double click to download “Models” folder 
    4. Move the “Models” folder to the “Organogenesis” folder
    5. Right click on Model folder and click “Extract Here”

        **Note:** This will replace the contents of the current “Models” folder. If you have already trained your own model, please remove it so that it is not overwritten. 

## Examples

Examples using each of the five functions are available below. Note that it is convenient to use bash scripts on Linux/MacOS or batch files on Windows for starting the program, as they can then be reused with the same parameters easily. The accompanying bash scripts and batch files for the examples below can be found in the "Example Scripts" folder of the main Organogenesis repository. 

### 1. Training a Model

**Required Arguments for Training a Model:**

* -o 
* -f Train
* --lr (learning rate)
* --epochs 
* --modelType

**Optional Arguments for Training a Model:**



* --processData 
* --dataPath 
* --loadModel
* --dataAugmentation
* --sortData
* --preSorted

All optional arguments are set to False or None by default. Supplying a boolean argument in a bash/batch script or in the command line sets the value to True. 

For example, the program could be used to process patient data and train a UNet model for predicting spinal cord contours using a learning rate of 0.001 and 10 epochs by entering the following commanding into the terminal/Command Prompt:

```
$ python Menu.py -o Spinal-Cord -f Train --lr 1e-3 --epochs 10 --modelType UNet --processData
```

If any of the required arguments are not supplied, then the user will be prompted to enter additional information. 

**Example bash script for Linux/MacOS** <br /> The following bash script (text file saved with a .sh extension) will start training a UNet model for the spinal cord using a learning rate of 0.001 for 10 epochs, where patient files are already in Patient_Files and data has not been processed into testing/validation/training folders. 

>#!/usr/bin/env bash <br /> chmod +x Menu.py <br/> ./Menu.py -o cord  <br/> \-f Train \ <br/> --lr 0.001 \  <br/> --epochs 10 \ <br/> --modelType UNet \ <br/> --processData

**Example batch file for Windows** <br /> The following batch file (text file saved with a .bat extension) will start training a UNet model for the spinal cord using a learning rate of 0.001 for 10 epochs, where patient files are already in Patient_Files and data has not been processed into testing/validation/training folders. 

>@ECHO OFF  <br /> python Menu.py -o cord ^  <br /> -f Train ^  <br /> --lr 0.001 ^  <br /> --epochs 10 ^  <br /> --modelType UNet ^ <br /> --processData  <br /> @pause

### 2. Generating Contours

**Required Arguments for Generating Contours:**



* -o (Can be a single organ, multiple organs separated by a single space, or “all” to generate predictions for all existing models)
* -f GetContours
* --predictionPatientName
* --modelType

**Optional Arguments for Generating Contours:**



* --thres 
* --contoursWithReal
* --loadContours
* --dontSaveContours
* --dataPath

All optional arguments are set to False or None by default. Supplying a boolean argument in a bash/batch script or in the command line sets the value to True. 

Here predictionPatientName is the FOLDER name for the patient which is to have contours generated. This patient can either be in the Patient_Files folder or the --dataPath argument should be supplied which points to the patient. 

Note that all models have a unique pixel threshold for predicting organs, and this threshold is determined with the BestThreshold function. If this function has been previously run then the program will automatically load this value (which is saved in the Model directory) and use it for predicting. If it has not been run, then the program will automatically run it before beginning predictions. This function can take 20-30 minutes. If threshold values are known and the user wishes to input what values to use regardless of the BestThreshold return value, then the --thres argument can be supplied with values for different organs separated by a single space.

As an example, to generate predictions for the spinal cord, brain stem and parotid glands, then the following command can be used to start the program:

```
$ python Menu.py -o spinal-cord brain-stem right-parotid left-parotid -f GetContours --predictionPatientName /path/to/predictionFileName 
```

**Example bash script for Linux/MacOS** <br /> If all of the possible organs have trained MultiResUNet models and are to be contoured then the following bash script could be used.

  

>#!/usr/bin/env bash <br /> chmod +x Menu.py <br /> ./Menu.py -o all \ <br /> -f GetContours \ <br /> --modelType MultiResUNet \ <br /> --predictionPatientName patientName \ 

**Example batch file for Windows** <br /> If all of the possible organs have trained MultiResUNet models, and are to be contoured then the following batch file could be used.

  

>@ECHO OFF <br /> python Menu.py -o all ^ <br /> -f GetContours ^ <br /> --modelType MultiResUNet ^ <br /> --predictionPatientName patientName <br /> @pause

### 3. Determining a Models Optimal Threshold Value

To determine a model’s optimal threshold for predicting whether a pixel is a part of an organ or not (by iterating through a series of values and determining which maximizes F Score) the BestThreshold function can be used. All that needs to be supplied as arguments are the organ of interest, the model type, and the BestThreshold function. Note that this function will be automatically called after a model completes the specified number of epochs during training. It will need to be run manually in the case that training is ended before all epochs have been completed. 

**Required Arguments for Determining the Optimal Threshold Value:**



* -o 
* -f BestThreshold
* --modelType

To determine the best threshold value for a UNet body model, then the program can be started with the command:

```
$ python Menu.py -o body -f BestThreshold -modelType unet
```

**Example bash script for Linux/MacOS** <br /> The following bash script can be used for determining the best threshold of a MultiResUNet spinal cord model.

>#!/usr/bin/env bash <br />
chmod +x Menu.py <br />
./Menu.py -o cord \ <br />
-f BestThreshold \ <br />
--modelType MultiResUNet

**Example batch file for Windows**  <br /> The following batch file can be used for determining the best threshold of a MultiResUNet spinal cord model.

>@ECHO OFF <br />
python Menu.py -o cord ^ <br />
-f BestThreshold ^ <br />
--modelType MultiResUNet  <br />
@pause

### 4. Getting Statistics for a Model

To generate a text file containing the hyperparameters of a model, the F Score, and the 95th percentile Haussdorff distance, the GetEvalData function is used. 

**Required Arguments for Getting Model Statistics:**



* -o 
* -f GetEvalData
* --modelType

**Optional Arguments for Getting Model Statistics:**



* --thres 

To generate statistics for a larynx UNet model, the program can be executed with the command 

```
$ python Menu.py -o larynx -f GetEvalData --modelType unet
```

**Example bash script for Linux/MacOS** <br/> To get the model statistics for a spinal cord UNet model with a known threshold value of 0.3, the following bash script may be used.

>#!/usr/bin/env bash <br/> 
chmod +x Menu.py <br/> 
./Menu.py -o cord \ <br/> 
--modelType UNet\ <br/> 
-f GetEvalData \ <br/> 
--thres 0.3

**Example batch file for Windows**  <br/> To get the model statistics for a spinal cord UNet model with a known threshold value of 0.3, the following batch file may be used. 

>@ECHO OFF  <br/> 
python Menu.py -o cord ^  <br/> 
-f GetEvalData ^  <br/>
--modelType UNet ^  <br/> 
--thres 0.3  <br/> 
@pause

### 5. Plotting Existing and Predicted 2D Masks

The function PlotMasks can be used to plot 2D CTs with both manually drawn and predicted masks for visual comparison. Patient CT images are chosen from the validation folder. 

**Required Arguments for Plotting Masks:**



* -o 
* -f PlotMasks
* --modelType

**Optional Arguments for Getting Model Statistics:**



* --thres 

Plots of manually contoured masks and masks predicted by a left parotid UNet model can be found using the command

```
 $ python Menu.py -o left-parotid -f PlotMasks --modelType UNet
```

**Example bash script for Linux/MacOS** <br /> To plot manually contoured masks and masks predicted by a brainstem MultiResUNet model with a known threshold value of 0.2, the following bash script may be used. 

>#!/usr/bin/env bash  <br /> 
chmod +x Menu.py <br /> 
./Menu.py -o brainstem \ <br /> 
-f PlotMasks \ <br /> 
--modelType MultiResUNet \ <br /> 
--thres 0.2 \ 

**Example batch file for Windows** <br /> To plot manually contoured masks and masks predicted by a brainstem MultiResUNet model with a known threshold value of 0.2, the following batch file may be used. 

>@ECHO OFF <br /> 
python Menu.py -o brainstem ^ <br /> 
-f PlotMasks ^ <br /> 
--modelType MultiResUNet ^  <br /> 
--thres 0.2 <br /> 
@pause

