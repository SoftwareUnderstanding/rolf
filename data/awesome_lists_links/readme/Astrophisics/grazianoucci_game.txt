# Game
*GAlaxy Machine learning for Emission lines*

[![Python version](https://img.shields.io/badge/Python-2.7-blue.svg)](https://www.python.org/download/releases/2.7.0/)

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/licenses/Apache-2.0) [![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/grazianoucci/game/issues)

## Table of content

- [Input files](#input-files)
- [Output files](#output-files)

# Input files

The program accepts three files with the following formats:

1. `Input file (line intensities, i.e. features for the ML algorithm)`
In the first row there are the wavelengths (units: Angstrom) of each line. Starting from the second row there are the emission lines intensities. Each row corresponds to a different input spectrum. Intensities for NON-DETECTIONS/MISSING-VALUES must be set equal to **0**. For example the input file format for three observed spectra with the H_alpha, [OIII]5007A and H_beta input lines should be:
```
6563 5007 4861
I_11 I_12 I_13
I_21 I_22 I_23
I_31 I_32  0
```
(N.B. I_* are the intensities of the lines, and for the sake of clarity, it has been assumed in the previous example a NON-DETECTION for the H_beta line in the third spectrum)

2. `Input file (uncertainties on line intensities)`
As in (1), in the first row there are the wavelengths (units: Angstrom) of each line. Starting from the second row there are the emission lines errors. Each row corresponds to a different input spectrum. Errors associated to NON-DETECTIONS/MISSING-VALUES must be set equal to **0**. A value in the input file of intensities is considered an UPPER LIMIT if the value for the corresponding uncertainty is set equal to **-99**. For example the error input file format for three observed spectra with the H_alpha, [OIII]5007A and H_beta lines should be:
```
6563 5007 4861
E_11 I_12 I_13
I_21  -99 I_23
I_31 I_32  0
```
(N.B. E_* are the uncertainties of the lines, and for the sake of clarity, it has been assumed in the previous example a NON-DETECTION for the H_beta line in the third spectrum and an UPPER LIMIT for the [OIII] line in the second spectrum.)

3. `Input file (labels)`
The user must choose in the file library/library_labels.dat the corresponding CLOUDY13.03 labels for the input line intensities. Each row contain a label. The total number of rows must be equal to the total number of observed lines (i.e. the total number of columns of the input files with line intensities). For example the input file containing the labels for three observed spectra with the H_alpha, [OIII]5007 and H_beta lines should be:
```
H-alpha 6563A
[O III] 5007A
H-beta 4861A
```

# Output files

Here there are reported the list of the output files produced by the program.
N.B. OPTIONAL files are computed only if the user answers "y" to the question "Do you want to create the optional files [y/n]?:"
N.B. The string "additional" in the name of the file refers to the physical properties "Av" and "fesc".

1. `output/model_ids.dat`
The code is able to deal with missing value (i.e. NON_DETECTIONS/MISSING-VALUES, set as zeros in the input file for the line intensities). In this file there are the details for each computed model.

2. `output/output_pdf_*.dat (OPTIONAL)`
For each input spectrum, the basic idea is to use a sort of bootstrap samples from the input data to build a set of new inputs (whose number is defined by the variable "n_repetition" inside the code) for the determination of physical properties. These new input data are constructed by perturbing the line intensities incorporating the errors on them (assuming that the errors are normally distributed). The code first generates multiple individual new observations of each spectrum that could subsequently be combined into a final PDF, which can be used to estimate the values for the physical properties and eventually the associated error. Each line within the file "output/output_pdf_*.dat" contains a list of different determinations of the physical properties for each input spectrum. Therefore the number of rows in these files are the total number of input models, the number of columns is instead the total number of determinations of the physical properties for each input model (defined by the variable "n_repetition" inside the code).

3. `output/output_ml.dat`
In this file there are the mean, median and standard deviation for the physical properties simply computed from each row of the files described in (2). The columns are:
```
- col1:  id_model
- col2:  mean[Log(G0)]
- col3:  median[Log(G0)]
- col4:  sigma[Log(G0)]
- col5:  mean[Log(n)]
- col6:  median[Log(n)]
- col7:  sigma[Log(n)]
- col8:  mean[Log(NH)]
- col9:  median[Log(NH)]
- col10: sigma[Log(NH)]
- col11: mean[Log(U)]
- col12: median[Log(U)]
- col13: sigma[Log(U)]
- col14: mean[Log(Z)]
- col15: median[Log(Z)]
- col16: sigma[Log(Z)]
```

4. `output/output_true_*.dat`, `output/output_pred_*.dat (OPTIONAL)`
The files "output_true_*.dat" contain the values for the physical parameters of a fraction equal to 10% of the library (defined by the variable "test_size" inside the code) and the files "output_pred_*.dat" describe the predicted values after applying the Machine Learning model to the testing dataset (10% of the library, defined by the variable "test_size" inside the code). With these files it is possible to compare the prediction of the models for a testing dataset in order to check the predictive performances. Each row in these files corresponds to a different "id_model" described in (1).

5. `/output/output_feature_importances.dat`
This writes down the output with the feature importances for each input line. Each row corresponds to a different "id_model" described in (1).
