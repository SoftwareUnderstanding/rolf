# MIDAS Task2

### This repository contains the code and documentation for MIDAS@IIITD Summer Internship/RA Task 2021 Task 2.

<hr>

A detailed explanation of this entire project is given in [MIDAS_Task2.pdf](./MIDAS_Task2.pdf). All results and explanations are present. If possible, please download it for better quality.

To reproduce this environment on the local system:

    pip install requirements.txt

Links to the models have been given in their respective cells. However, the list of models is included in [this](./models_path.md) file. 

### This repository allows you to execute:

1. [Python Files](https://github.com/Saloni1Parekh609/MIDAS_Task2/tree/main/ExecutableFiles)

2. [Python Notebooks](https://github.com/Saloni1Parekh609/MIDAS_Task2/tree/main/Notebooks)

**Note: Please make sure to change the file paths accordingly.**

### Folder Structure:

```
├── ExecutableFiles
│   ├── Evaluation
│   ├── Point1
│   ├── Point2
│   └── Point3
└── Notebooks
    ├── DatasetCreation
    ├── Evaluation
    ├── Experiments
    ├── Point1
    ├── Point2
    └── Point3
```

### Results:

|         | Data Split(Train/Valid) | # of Epochs | Convergence Time  | Train Accuracy(%) | Valid Accuracy(%) | Tes Accuracy(%) |
| :-----: | :---------------------: | :---------: | :---------------: | :---------------: | :---------------: | :-------------: |
| Model F |        1860/620         |     100     |         -         |       88.65       |       75.64       |        -        |
| Model A |        1860/620         |     30      |   16.96 minutes   |       93.99       |       77.90       |        -        |
| Model B |       50000/10000       |     20      | 1 hour 43 minutes |       99.85       |       99.15       |      99.15      |
| Model C |       50000/10000       |     20      | 1 hour 40 minutes |       99.64       |       98.71       |      99.00      |
| Model D |       45000/15000       |     20      |    34 minutes     |       99.06       |       99.07       |      99.00      |
| Model E |       45000/15000       |     20      |    42 minutes     |       99.39       |       99.23       |      99.00      |

<hr>
### You can also execute this project via Google Colab and access resources from this [Drive](https://drive.google.com/drive/folders/18ZBFPV60wpDRyeerVIH47WZVOUiKJ5IT?usp=sharing) folder. Please add this folder’s shortcut to your drive before going ahead.

### References

End-to-End Object Detection with Transformers by FacebookAI: https://arxiv.org/pdf/2005.12872.pdf

