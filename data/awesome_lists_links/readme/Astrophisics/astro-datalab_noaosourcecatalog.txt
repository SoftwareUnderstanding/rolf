# NOAO Source Catalog

Software to create a source catalog of all NOAO imaging data.

# Processing Steps

Here's the current thinking on the processing steps, where each step is run over the entirety of the data
(with a wrapper) before the next step is started.

## 1) Measurement (loop over exposure)

    - nsc_instcal_main.pro: wrapper
    - nsc_instcal.py: core program, runs on one exposure

## 2) Calibration & QA metrics (loop over exposure)

    - nsc_instcal_calibrate_main.pro: wrapper
    - nsc_instcal_calibrate.pro: core program, runs on one exposure

## 3) Enforce QA cuts, cross-matching, averaging (loop over healpix)

    - nsc_instcal_combine_main.pro: wrapper
    - nsc_instcal_combine.pro: core program, runs on one Healpix pixel

## 4) Load database (loop over exposure and healpix)