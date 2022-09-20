# Nigraha: DL pipeline for finding planet candidates from TESS data

## Background
This repository contains the code, models, and data from our Nigraha pipeline for finding planet candidates from TESS.  [Here](https://arxiv.org/abs/2101.09227) is the paper that describes this work.  Our paper has been accepted for publication in MNRAS (Monthly Notices of Royal Astronomy Society journal).

## Citation
If you use this code, please cite it:

Rao, S & Mahabal, A & Rao, N & Raghavendra, C (2020).  Nigraha: Machine-learning based pipeline to identify and evaluate planet candidates from TESS. 
Monthly Notices of the Royal Astronomical Society, Volume 502, Issue 2, April 2021, Pages 2845–2858


## Required packages
    pip install transitleastsquares lightkurve matrixprofile tensorflow numpy pandas scipy

## Overview
The pipeline has 4 stages as described in the paper:

1. Period finding using Transit Least Squares (TLS) package.  This run on a sector by sector basis to build a per-sector catalog.
1. Transform the flux values in `.fits` lightcurve files to global/local views and write out the output in `.tfRecords` files.
1. Build a model on training data and save a checkpoint.
1. Load a previously saved model to generate predictions for new sectors.

In the repository, we provide candidates for several sectors.  For new sectors that you want to try the pipeline on, we provide helper scripts that can be run, and you can do your own analysis.  We describe below how you can run the pipeline to generate candidates.  The example described below is to generate predictions for Sector 14.  If you want to generate prediction for another script, please modify the scripts listed below (for example, for `Sector 32` replace `14` with `32`).

## Running the pipeline
Setup up the `PYTHONPATH` environment variable.

    # pwd = /Users/sriram/Astro/Nigraha
    source path_setup.sh
    
### Period finding
Download lightcurve files from MAST.  See `lcs/README.txt`.  Suppose that we have downloaded data for `Sector 14`. Run the following script to build the catalog.
 
    # pwd = /Users/sriram/Astro/Nigraha/utils
    sh build_catalog.sh
    
The output of this step is a file `catalog/period_info-sec14.csv`.  

### Generating Pipeline input
The downloaded lightcurves need to be pre-processed to generate TFRecords which are then input to the DL model. Run the following script to build the TFRecords.
    
    # pwd = /Users/sriram/Astro/Nigraha/data
    sh build_eval_data.sh
    
The output of this step are files in the folder `TFRecords/predict/sec14`.

### Generating predictions
The repository contains a trained model.  The model weights are stored in `models/weights/...`. Run the following script to generate predictions.
    
    # pwd = /Users/sriram/Astro/Nigraha/models
    sh gen_predict.sh

The output of this step is scores for TCEs.  The output is in `output/scores/sec14.csv`.  

Lastly, to federate with TOI catalog to identify known candidates and generate additional meta-data, run the following script.
    
    # pwd = /Users/sriram/Astro/Nigraha/utils
    sh run_gen_tce.sh

The output is in `output/candidates/sec14.csv`.  

### Building your own model
If you want to train a new model with other labeled datasets, modify and run the following script to build the training data:

    # pwd = /Users/sriram/Astro/Nigraha/data
    sh build_train_data.sh
    
Run the trainer and save the model weights:

    # pwd = /Users/sriram/Astro/Nigraha/models
    sh ensemble_train.sh

After this point, follow the steps to prepare input and generate predictions.
