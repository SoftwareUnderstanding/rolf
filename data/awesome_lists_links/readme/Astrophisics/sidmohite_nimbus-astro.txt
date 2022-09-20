# nimbus - A Bayesian inference framework to constrain kilonova models
nimbus is a hierarchical Bayesian framework to infer the intrinsic luminosity parameters
of kilonovae (KNe) associated with gravitational-wave (GW) events, based purely on non-detections.
This framework makes use of GW 3-D distance information and electromagnetic upper limits from
a given survey for multiple events, and self-consistently accounts for finite sky-coverage and 
probability of astrophysical origin.

## Installation
nimbus can be installed using the following commands:

    git clone git@github.com:sidmohite/nimbus-astro.git
    
    cd nimbus-astro
    
    pip install -r requirements.txt
    
    python setup.py install
    
Note : It is recommended that installation be done into a virtual Python/Anaconda environment with
python >=3.6 and <4. This package requires `astropy`, `healpy`,`numpy`, `pandas`, `scipy` and a recent
version of `setuptools` (currently fixed to version `57.1.0` here).

## Data Inputs
In order to use nimbus to constrain kilonova models we first need to ensure we have the relevant
data files and that they have specfic attributes that will be used by the code.
**Note : The data formats and code presented in this repository are based on observations from the
Zwicky Transient Facility (ZTF). However, the code can be easily modified to account for other surveys.** 

* A `survey file` containing field, pixel and extinction specific information for the survey.
    * Currently the code expects this file to exist as a Python pickle  - `.pkl` file.
    * The file should contain 3 attributes/columns at minimum -
        * `field_ID` - ID specifying which field.
        * `ebv` - E(B-V) extinction value along the line-of-sight of the field.
        * `A_lambda` - the total extinction in a given passband lambda.

* A `data file` containing all observational data for the event(s) from the survey including upper limits for each 
observed field and passband filter as well as associated observation times.
    * The code expects this file to be in `csv` or `txt` format.
    * The file should contain the following attributes/columns at minimum -
        * `jd` - Time of each observation (format : `isot`).
        * `scimaglim` - Limiting magnitude in the science image (at each CCD/observation point) for each observation.
        * `field` - Field ID (integer) labelling the field for the corresponding observation.
        * `fid` - Filter ID (integer) labelling the passband filter used for the corresponding observation. The code uses
                  the following convention for the 3 passbands of ZTF : `fid` - '1' : `g`, '2' : `r`, '3' : `i`
        * `status` - Status number (integer) indicating if the observation is good to use for the analysis. Convention :
          `status=1` refers to a "good" observation. 
    
* A `skymap file` containing the 3-D GW skymap localization information.
    * The code expects this to be in `fits.gz` format (identical to that released by LIGO for public alerts).

* A `sample_file` containing prior hyperparameter samples drawn according to a assumed prior distribution.
    * The code expects this file to have a `csv` or `txt` format.

## Running Executables
In order to perform the inference, we need to run the executables given in this package in a specific order:

### Step 1 : Single-field inference
To find the likelihood for the data in a given field given the hyperparameter samples, we can make use of the 
`singlefield_calc` [executable](https://github.com/sidmohite/nimbus-astro/blob/master/nimbus/singlefield_calc).
Since a survey would, in general, contain a large number of observed fields with associated data, it is ideal to 
run this executable with parallelized instances on a computing framework such as a high performance cluster.

    Usage: singlefield_calc [options]

    Options:
      -h, --help            show this help message and exit
      --field=FIELD         Field number to calculate the likelihood for.
      --data_file=DATA_FILE
                            File containing all observational data for the event
                            from the survey.
      --survey_file=SURVEY_FILE
                            File containing field, pixel and extinction specific
                            information for the survey.
      --skymap_file=SKYMAP_FILE
                            Skymap file for the event.
      --sample_file=SAMPLE_FILE
                            File containing the points in parameter space to
                            calculate the log-posterior for.
      --t_start=T_START     The start time of the data for the event. Format must
                            be in isot.
      --t_end=T_END         The end time of the data for the event. Format must be
                            in isot.
      --single_band         Indicator that makes the analysis a single-band
                            calculation.
      --output_str=OUTPUT_STR
                            The common string pattern for the files that save the
                            likelhood values for each field.
                            
### Step 2 : Compute field probabilities
The next step in the inference involves computing the overall probability of the kilonova event being localized
within each field for which we ran [Step 1](https://github.com/sidmohite/nimbus-astro/blob/master/README.md#step-1--single-field-inference)
above. This is calculated from the survey and skymap files provided for the event, using the `compute_field_probs` [executable](https://github.com/sidmohite/nimbus-astro/blob/master/nimbus/compute_field_probs) and is used in the generation of the posterior values for
each hyperparameter sample when we combine field likelihoods in [Step 3](https://github.com/sidmohite/nimbus-astro/blob/master/README.md#step-3--combine-field-likelihoods).

    usage: compute_field_probs [-h] --field_probs_file FIELD_PROB_FILE
                           --survey_file SURVEY_FILE --skymap_file SKYMAP_FILE
                           --infield_likelihoods_path INFIELD_LIKELIHOODS_PATH
                           --common_str COMMON_STR

    Calculate the field probabilities and store them in file.

    optional arguments:
      -h, --help            show this help message and exit
      --field_probs_file FIELD_PROB_FILE
                            File to save the field probabilities in.
      --survey_file SURVEY_FILE
                            File containing field, pixel and extinction specific
                            information for the survey.
      --skymap_file SKYMAP_FILE
                            Skymap file for the event.
      --infield_likelihoods_path INFIELD_LIKELIHOODS_PATH
                            Path to files containing sample likelihood values for
                            each field. The code expects files to be named using a
                            common string pattern (see below) with the field
                            number appended at the end.
      --common_str COMMON_STR
                            The common string pattern (see below) the code expects
                            the files to be named with.
                           
### Step 3 : Combine field likelihoods
The final step in the inference is to combine the individual field likelihoods and field probabilites from Steps 1 and 2
to give us the log-posterior values for each hyperparameter sample. This is done using the `combine_fields` [executable](https://github.com/sidmohite/nimbus-astro/blob/master/nimbus/combine_fields).

    usage: combine_fields [-h] --sample_file SAMPLE_FILE --field_probs_file
                      FIELD_PROB_FILE --infield_likelihoods_str
                      INFIELD_LIKELIHOODS_STR [--coverage_fraction COV_FRAC]
                      --P_A P_ASTRO --output_file OUTPUT_FILE

    Combine the in-field likelihoods to construct the final posterior

    optional arguments:
      -h, --help            show this help message and exit
      --sample_file SAMPLE_FILE
                            File containing sample points.
      --field_probs_file FIELD_PROB_FILE
                            File containing the total sky probability for each
                            field.
      --infield_likelihoods_str INFIELD_LIKELIHOODS_STR
                            Common string for files containing sample likelihood values for
                            each field. The code expects files to be named using a
                            common string pattern (see below) with the field
                            number appended at the end.
      --coverage_fraction COV_FRAC
                            Assumed pseudo fraction of the event skymap that is
                            surveyed by the telescope. Range(0-1) (default: 0)
      --P_A P_ASTRO         Probability of the event being
                            astrophysical.Range(0-1).
      --output_file OUTPUT_FILE
                            Output file.

The final data product is a file containing the hyperparameter samples and correpsonding log-posterior values.

An example jupyter notebook demonstrating the basic inference and associated data is provided in the 
`examples/` directory in this repository.

## Citation
If you use this package in a publication please cite the paper [Mohite et al. (2021)](https://arxiv.org/abs/2107.07129) and package on Zenodo
    <a href="https://zenodo.org/badge/latestdoi/374686703"><img src="https://zenodo.org/badge/374686703.svg" alt="DOI"></a>
