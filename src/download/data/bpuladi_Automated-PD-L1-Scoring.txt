# Automated PD-L1 Scoring
These files are part of a scientific paper and describe a workflow for automated PD-L1 assessment of Tumor Proportion Score (TPS), Combined Positive Score (CPS), and Immune Cell Score (ICS) with QuPath and MATLAB exemplified with head and neck squamous cell carcinoma (HNSCC). 

> Puladi, B.; Ooms, M.; Kintsler, S.; Houschyar, K.S.; Steib, F.; Modabber, A.; Hölzle, F.; Knüchel-Clarke, R.; Braunschweig, T. Automated PD-L1 Scoring Using Artificial Intelligence in Head and Neck Squamous Cell Carcinoma. Cancers 2021, 13, 4409. https://doi.org/10.3390/cancers13174409

**NOT FOR DIAGNOSTIC USE!**

## Training data
HNSCC cases from the [National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC)](https://wiki.cancerimagingarchive.net/display/Public/CPTAC-HNSCC) were used to train the following models.

- Training data can be exported class by class from annotations in QuPath as tile images: [exportTiles.groovy](/training/exportTiles.groovy)
- From HE training data, the hematoxylin channel can be extracted using MATLAB script: [deconvolveTiles.m](/training/deconvolveTiles.m)

## Pre-trained models / Training new models

Train models for automated PD-L1 scoring 

1. First neural network:
    - The first neural network is designed to annotate the tumor. For this purpose shufflenet [1] was used and trained on 8 tissue classes using MATLAB: [trainFirstNetwork.m](/training/trainFirstNetwork.m)
    - A pre-trained model "shufflenet-HNSCC-NumClass-8.mat" for HNSCC can be found under the folder [models](/models/).
2. Second neural network:
    - The second neural network is designed to detect the cells.
    - StarDist [2] is used for cell recognition. A pre-trained model can be used for this purpose: [he_heavy_augment.zip](https://github.com/stardist/stardist-imagej/tree/master/src/main/resources/models/2D/he_heavy_augment.zip)
3. Third neural network:
    - The third neural network is designed to classify cells into tumor, immune and stromal cells.
    - Training tiles and test tiles must be created for the third neural network. This can be done with the following script in MATLAB: [mergeTiles.m](/training/mergeTiles.m)
    - Afterwards the merged tile images can be imported into QuPath and the cells can be recognized using the following script: [step2_detectCells.groovy](workflow/step2_detectCells.groovy)
    - By means of the function "Classify &#8594; Object classification &#8594; Train object classifier" a new classifier can be trained in QuPath.
    - The trained classifier can then be evaluated on another dataset with merged tile images. A helpful Groovy script for this: [validateMLP.groovy](training/validateMLP.groovy)
    - A pre-trained model "HNSCC_MLP_LYM_TUM_STR.json" for HNSCC can be found under the folder [models](/models/).

## Workflow
1. Annotate tumor:
    - Create labelmap using pre-trained model in MATLAB: [step1_1_annotateTumor.m](workflow/step1_1_annotateTumor.m)
    - Create project and import WSIs + models (see dependency + models) + import labelmap
    - Prepare WSI with the  following script: [step_1_2_estimateBackgroundValues.groovy](dependency/step_1_2_estimateBackgroundValues.groovy)
    - Import labelmap as an annotation into QuPath with the following script: [step_1_3_importBinaryLabelmaps.groovy](dependency/step_1_3_importBinaryLabelmaps.groovy)
2. Detect cells:
    - Detect all cells inside annotations with the following script in QuPath: [step2_detectCells.groovy](workflow/step2_detectCells.groovy)
3. Classify cells:
    - Run classifier with the following script in QuPath: [step3_classifyCells.groovy](workflow/step3_classifyCells.groovy)
4. Calculate PD-L1 scores:
    - Calculate PD-L1 Scores with the following script in QuPath: [step4_calculateScores.groovy](workflow/step4_calculateScores.groovy)

## Used software
- [QuPath version 0.2.3](https://github.com/qupath/qupath)
- [MATLAB 2021a](https://www.mathworks.com)

## License
[GNU General Public License v2.0](/LICENSE)

## Reference
1. Zhang, X.; Zhou, X.; Lin, M.; Sun, J. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, 2017. Available online: https://arxiv.org/pdf/1707.01083.
2. Schmidt, U.; Weigert, M.; Broaddus, C.; Myers, G. Cell Detection with Star-convex Polygons 2018, 11071, 265–273, doi:10.1007/978-3-030-00934-2_30.
3. Bankhead, P.; Loughrey, M.B.; Fernández, J.A.; Dombrowski, Y.; McArt, D.G.; Dunne, P.D.; McQuaid, S.; Gray, R.T.; Murray, L.J.; Coleman, H.G.; et al. QuPath: Open source software for digital pathology image analysis. Sci. Rep. 2017, 7, 16878, doi:10.1038/s41598-017-17204-5.
4. The MathWorks, Inc. Deep Learning Toolbox. Available online: https://www.mathworks.com/help/deeplearning/
