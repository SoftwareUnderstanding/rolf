This repository contains code used for the paper "A Benchmark of Medical Out of Distribution Detection."[ArXiv](https://arxiv.org/abs/2007.04250).

Code is based on the repository from "Does Your Model Know the Digit 6 Is Not a Cat? A Less Biased Evaluation of Outlier Detectors." [ArXiv](https://arxiv.org/abs/1809.04729).

This code is provided "as-is" and is not guaranteed to work out-of-the-box.
# Datasets and methods
Our additions include:
1. Datasets:
    * ANHIR: Automatic Non-rigid Histological Image Registration Challenge [link](https://anhir.grand-challenge.org/).
    * DRD: High-resolution retina images with presence of diabetic retinopathy in each image labeled on a scale of 0 to 4. We convert this into a classification task where 0 corresponds to healthy and 1-4 corresponds to unhealthy.  [link](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
    * DRIMDB: Fundus images of various qualities labeled as good/bad/outlier. We use the images labeled as bad/outlier in evaluation 3, use-case 2.
    * Malaria Image of cells in blood smear microscopy collected from healthy persons and patients with malaria. Used in evaluation 4 use-case 1.[link](https://lhncbc.nlm.nih.gov/publication/pub9932)
    * MURA: MUsculoskeletal RAdiographs is a large dataset of skeletal X-rays. We use its validation split in evaluation 1 and 2's use-case 1. Images are grayscale and the square cropped.  [link](https://stanfordmlgroup.github.io/competitions/mura/) 
    * NIH Chest: This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with 14 condition labels. The x-rays images are in posterior-anterior view (X-tray traverses back to front).  [link](https://www.kaggle.com/nih-chest-xrays/data)
    * PAD Chest: This is a large scale chest X-ray dataset. It is labeled with 117 radiological findings - we use the subset with correspondence to the 14 condition labels in the NIH Chest dataset. Images are in 5 different views: posterior-anterior (PA), anterior-posterior (AP), lateral, AP horizontal, and pediatric. [link](https://bimcv.cipf.es/bimcv-projects/padchest/)
    * PCAM: Patch Camelyon dataset is composed of histopathologic scans of lymph node sections. Images are labeled for presence of cancerous tissue. [link](https://github.com/basveeling/pcam)
    * RIGA: Fundus imaging dataset for glaucoma analysis. Images are marked by physicians for regions of disease. We use this dataset for evaluation 3, use-case 3.
2. OoD Detection Methods:
    * ALI + Reconstruction Threshold: uses Adversarially Learning Inference [link](https://arxiv.org/abs/1606.00704) to train auto-encoder.
    * Mahalanobis: Uses Gaussian discriminant analysis in classifier feature space to distinguish In/Out of distribution [link](https://arxiv.org/abs/1807.03888).
     
# Code Structure
We largely kept the same code structure as [OD-test](https://github.com/ashafaei/OD-test) with the following additions:
1. In `preproc` are code for preprocessing some medical datasets. High resolution images are converted to 224x244 resolution, and images with useful labels are selected. 
2. In `setup` are code for training NNs on source datasets (DRD, NIH Chest, PAD Chest, PCAM). Default hyperparameters are used.
3. `[IN_dataset_name]_eval_rand_seeds.py` are main scripts for evaluating OD methods on datasets. Some OD methods may be commented out and should be uncommented in the `__main__` block. `methods_64` are methods that uses 64x64 resolution, while `methods` use 224x224 resolution.
