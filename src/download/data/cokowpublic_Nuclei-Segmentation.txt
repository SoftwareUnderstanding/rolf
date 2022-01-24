# Nuclei-Segmentation

Goal: To design a single algorithm that is capable of both object detection and instance segmentation for nuclei seen in imaging.
As the nuclei are varied in size, magnification, modality, etc, one is forced to resort to modern methods in feature extraction
and analysis with special attention required on spatial resolutions and generalization (especially considering the numerous errors
present in the training data labels).

Model currently has two implementations.  Both involve significant image pre-processing that feeds into 1) a Sobel filtering method and 2) Mask R-CNNs (as outlined in https://arxiv.org/pdf/1703.06870.pdf).


* Currently in the process of migrating files and forks over to this repo, including visualization, classical attempts at segmentation, 
model evaluation,and leading attempt(s) (after necessary passing of first project deadline). *
