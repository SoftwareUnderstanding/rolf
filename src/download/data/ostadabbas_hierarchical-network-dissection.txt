# Hierarchical Network Dissection

This is the official pytorch implementation of the Hierarchical Network Dissection which performs network dissection on several face models as described in the [paper](https://arxiv.org/pdf/2108.10360.pdf). Also, this repo contains the link to the first ever Face Dictionary that contains several face concepts annotated under the same dataset.

## Contents

* [Prerequisites](#prerequisites)
* [Preparation](#preparation)
* [How to Run](#how-to-run)

    * [Stage 1 (Dissection)](#stage-1-(dissection))
    * [Stage 2 (Probabilistic Hierarchy)](#stage-2-(probabilistic-hierarchy))
    * [Stage 3 (Global Bias)](#stage-3-(global-bias))

* [Acknowledgement](#acknowledgement)


## Prerequisites:

- Torch=1.5.0
- PIL
- opencv
- numpy
- matplotlib
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- cuda/cudnn (recommended)
- tqdm

## Preparation:

- **Data**
    - Download the data [here](https://coe.northeastern.edu/Research/AClab/Face_Dictionary/) and put in visual_dictionary folder.
    - Download the models from [here](https://coe.northeastern.edu/Research/AClab/Face_Dictionary/) and put them in model folder.
- **Code**
    - Change flags in settings.py to determine which model to dissect.
    - Change the name of the layer in model loader scripts to choose which layer to dissect.


## How to run:

### Stage 1 (Global Bias)

In order to determine if any of the units in the dissected layer has a bias towards any of the 4 global concepts we have provided in the dictionary, we generate feature files per model for each concept. Then these feature files are analyzed to generate a set of probabilities per subgroup and these probabilities are saved in both pickle and text format for reusability and readability respectively with the following names in the 'Non Localizable' folder which is defined in the settings file.

- layer_model_bias(skin_tone).npz
- layer_model_bias(ethnic).npz
- layer_model_bias(gender).npz
- layer_model_bias(age).npz
- layer_nl_probs.txt
- layer_nl_probs.pkl

The individual probabilities of the models can be plotted by setting plot = True in `bias_analysis()` function in `nl_bias.py` script. Apart from that each global concept has a plot comparing the number of biased units per subgroup for all dissected models.

### Stage 2 (Network Dissection)
The first stage of the local concept interpretation process requires us to extract the activation maps of all the images (N) in the Face Dictionary with labelled local concepts for the layer specified in the chosen model loader script accordingly. For all the units (U) in the given layer, we generate N maps from the `main.py` script which first stores the activations in a numpy memory map. The size of the memory map is smaller for the deeper layers due to the lower resolutions of the activation map (HxW). The first function 'feature_extraction' generates two files and stores them in the auto generated 'result' folder that are:

- layer.mmap
- layer_feature_size.npy

Secondly after generating the features, we now estimate the threshold values per unit by using their spatial features and computing a value such that the probability of any spatial location having a value greater than the threshold is equal to 0.005 (99.5 quantile). The second function called 'quantile_threshold' computes that and these values are stored in a file called:

- layer_quantile.npy

Finally, we use these thresholds to segment the activation maps for each unit respectively and evaluate them against the binary labels we have assembled in the dictionary per concept to compute their intersections and unions with them. Only those images that have a labelled instance of a given concept are used for this evaluation and once we iterate through the entire dictionary, we obtain a list of intersections and unions for each unit that has a length eqaul to the number of concepts. Then we divide the intersections by the unions and generate a final dataset wide IoU for each unit-concept pair. The concept with the highest IoU is recorded and the unit is said to be interpretable if the top concept has an IoU > 0.04. The third function 'tally' performs this computation and generates a pdf file (as shown below) that displays the top four images with the highest IoU returned for that concept per unit. This function generates two files that record the IoU values and display the dissection report respectively which are:
- layer_tally.npz
- layer_conv.pdf

![Alt text](https://i.postimg.cc/bYVNCHn4/report-photo.png)

### Stage 3 (Probabilistic Hierarchy)

Even though we can obtain the dominant concepts per unit based on IoU, more often than not there is more than one concept that manages to obtain a high IoU and it is very likely these concepts lie in a similar region of the face. In that case, it is better to establish a hierarchy of concepts that lie in the same region of the face as that of the top concept returned by Stage 1 pipeline. In order to do that we run the `cluster_top.py` script as it identifies the facial region and then generates probabilities for every concept within that region of the face. This script identifies all the interpretable concepts in that region and plots the histogram for all local concepts detected by each model and saves their tally as following:

- layer_unit_concept_probs.pkl
- layer_unit_concept_probs.txt

The individual histograms per model as well as the comparison plots for regional affinity and different concept types are also generated by this script.

![Alt text](https://i.postimg.cc/tRcptRQj/Cluster-probs.png)

## Acknowledgement

Thank you for the original Network Dissection implementation

- [Network Dissection Lite in PyTorch](https://github.com/CSAILVision/NetDissect-Lite)

## License

- This code is for non-commercial purpose only.
- For further inquiry please contact -- Augmented Cognition Lab at Northeastern University [here](http://www.northeastern.edu/ostadabbas/)
