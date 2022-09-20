## FIRST Classifier: Compact and Extended Radio Galaxies Classification using Deep Convolutional Neural Networks
- ### Wathela Alhassan et al.
### Paper
MNRAS: https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/sty2038/5060783


Astro-ph: https://arxiv.org/abs/1807.10380


### [FIRST Classifier](FIRST_CLASSIFIER.py)
We present the FIRST Classifier, an on-line system for automated classification of Compact and Extended radio sources. We developed the FIRST Classifier based on a trained Deep Convolutional Neural Network Model to automate the morphological classification of compact and extended radio sources observed in the FIRST radio survey. Our model achieved an overall accuracy of 97% and a recall of 98%, 100%, 98% and 93% for Compact, BENT, FRI and FRII galaxies respectively. The current version of the FIRST classifier is able to predict the morphological class for a single source or for a list of sources as Compact or Extended (FRI, FRII and BENT).

<img src="https://github.com/wathela/FIRST-CLASSIFIER/blob/master/Diagram.png" width=493px>

### [Single Source classification](single_source_classification.py):
Classify only one single source.
- #### Input: 
  Coordinates of single radio source (Right Ascention and Declination in degree).
- #### Output: 
  Predicted morphology type(corresponding to the highest probability), probabilities plot of the classification and a direct link to download the FITS file cut out of the target.

- How to run example:
  `python3 single_source_classification.py`
  then you will be asked to inter the Ra and Dec of the source.
### [Multi Sources classification](multi_sources_classification.py):
Allow the classification of a list of sources (csv file).
- #### Input: 
  A csv file that has a list of coordinates of sources and index of the Right Ascention and Declination columns ( RA and DEC must be in degree).
- #### Output: 
  A csv file containing 4 columns: Coordinates (RA and DEC), Predicted class, Highest probability, Link to download the cut-out FITS file.

- How to run example:

  `python3 multi_sources_classification.py --data_dir wathela/test.csv --ra_col 0 --dec_col 1`
### Requirement:
- #### Python 3.x with the [Required Packages](requirements.txt) installed.

### How to cite:
@article{Alhassan2018,

author = {Alhassan, Wathela and Taylor, A R and Vaccari, Mattia},

doi = {10.1093/mnras/sty2038},

issn = {0035-8711},

journal = {Monthly Notices of the Royal Astronomical Society},

month = {jul},

title = {{The FIRST Classifier: Compact and Extended Radio Galaxy Classification using Deep Convolutional Neural Networks}},

url = {https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/sty2038/5060783},

year = {2018}

}
